# ---------------------------------------------------------------------------
# YOLOv8n Optic Disc Localizer

import os
import cv2
import numpy as np
import shutil
import yaml
import mlflow
import torch
from tqdm import tqdm
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE        = os.getenv("REFUGE2_BASE", "/kaggle/input/datasets/victorlemosml/refuge2/REFUGE2")
YOLO_DIR    = os.getenv("YOLO_DIR",     "/kaggle/working/yolo_dataset")
RUNS_DIR    = os.getenv("RUNS_DIR",     "/kaggle/working/yolo_runs")
MODEL_DIR   = os.getenv("MODEL_DIR",    "/kaggle/working/yolo_model")
EPOCHS      = 50
IMGSZ       = 640
BATCH       = 16

# ---------------------------------------------------------------------------
# Dagshub + MLflow — set BEFORE everything else so YOLO autologging
# picks up the Dagshub URI and logs live during training
# ---------------------------------------------------------------------------
os.environ["MLFLOW_TRACKING_URI"]      = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Create/set experiment so YOLO logs into a named experiment
mlflow.set_experiment("YOLOv8n_DiscLocalizer")
print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"✓ Experiment: YOLOv8n_DiscLocalizer")


# ---------------------------------------------------------------------------
# Step 1 — Generate YOLO labels from disc masks
# ---------------------------------------------------------------------------

def mask_to_yolo_bbox(mask_path):
    """
    Reads a REFUGE2 mask and returns YOLO format bbox.
    REFUGE2 mask: pixel <= 128 = optic disc, pixel == 0 = optic cup
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    disc   = (mask <= 128).astype(np.uint8)
    ys, xs = np.where(disc > 0)
    if len(ys) == 0:
        return None

    h, w = mask.shape

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 20% padding around disc
    pad_x = int((x_max - x_min) * 0.20)
    pad_y = int((y_max - y_min) * 0.20)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    bbox_w   = (x_max - x_min) / w
    bbox_h   = (y_max - y_min) / h

    return x_center, y_center, bbox_w, bbox_h


def generate_yolo_dataset(base, yolo_dir):
    print("Generating YOLO dataset from REFUGE2 masks...")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    total_generated = 0
    total_skipped   = 0

    for split in ["train", "val"]:
        img_dir  = os.path.join(base, split, "images")
        mask_dir = os.path.join(base, split, "mask")

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            print(f"  Skipping {split} — directory not found")
            continue

        img_files = sorted(os.listdir(img_dir))
        print(f"  Processing {split}: {len(img_files)} images")

        for img_file in tqdm(img_files, desc=split):
            stem     = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)

            mask_path = None
            for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif"]:
                cand = os.path.join(mask_dir, stem + ext)
                if os.path.exists(cand):
                    mask_path = cand
                    break

            if mask_path is None:
                total_skipped += 1
                continue

            bbox = mask_to_yolo_bbox(mask_path)
            if bbox is None:
                total_skipped += 1
                continue

            x_center, y_center, bbox_w, bbox_h = bbox

            shutil.copy2(img_path, os.path.join(yolo_dir, "images", split, img_file))

            label_path = os.path.join(yolo_dir, "labels", split, stem + ".txt")
            with open(label_path, "w") as f:
                f.write(
                    f"0 {x_center:.6f} {y_center:.6f} "
                    f"{bbox_w:.6f} {bbox_h:.6f}\n"
                )
            total_generated += 1

    print(f"\n✓ Generated {total_generated} label files")
    print(f"  Skipped {total_skipped} files")
    return total_generated


def write_yolo_yaml(yolo_dir):
    config = {
        "path":  yolo_dir,
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["optic_disc"]
    }
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✓ Dataset YAML written to {yaml_path}")
    return yaml_path


# ---------------------------------------------------------------------------
# Step 2 — Train YOLOv8n with live Dagshub MLflow logging
# ---------------------------------------------------------------------------

def train_yolo(yaml_path, runs_dir):
    print("\nInitializing YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training — logs streaming to Dagshub live...")
    print(f"View at: https://dagshub.com/jani2904-sys/NAION-Risk-Analyzer.mlflow\n")

    results = model.train(
        data       = yaml_path,
        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = BATCH,
        project    = runs_dir,
        name       = "disc_localizer",
        device     = 0 if torch.cuda.is_available() else "cpu",
        patience   = 10,
        save       = True,
        plots      = True,
        verbose    = True,
        cache      = True,        # cache images in RAM after epoch 1 — faster
        workers    = 2,           # reduce workers to avoid memory issues
        # Augmentations
        fliplr     = 0.5,
        flipud     = 0.3,
        degrees    = 15.0,
        hsv_h      = 0.015,
        hsv_s      = 0.3,
        hsv_v      = 0.2,
        scale      = 0.3,
        translate  = 0.1,
    )

    return model, results


# ---------------------------------------------------------------------------
# Step 3 — Evaluate
# ---------------------------------------------------------------------------

def evaluate_model(model, yaml_path):
    print("\nEvaluating on validation set...")
    metrics   = model.val(data=yaml_path)
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)

    print(f"  mAP@50:    {map50:.4f}")
    print(f"  mAP@50-95: {map50_95:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    return {
        "mAP50":     map50,
        "mAP50_95":  map50_95,
        "precision": precision,
        "recall":    recall,
    }


# ---------------------------------------------------------------------------
# Step 4 — Log final artifacts + best.pt to Dagshub
# ---------------------------------------------------------------------------

def log_final_to_mlflow(model, metrics, runs_dir):
    """
    YOLO autologging already streamed epoch metrics to Dagshub live.
    This step logs final summary metrics + best.pt artifact to the
    same experiment so everything is in one place.
    """
    print("\nLogging final artifacts to Dagshub MLflow...")

    best_pt = os.path.join(runs_dir, "disc_localizer", "weights", "best.pt")

    with mlflow.start_run(run_name="YOLOv8n_DiscLocalizer_Final"):

        mlflow.log_param("model",       "yolov8n")
        mlflow.log_param("epochs",      EPOCHS)
        mlflow.log_param("imgsz",       IMGSZ)
        mlflow.log_param("batch",       BATCH)
        mlflow.log_param("task",        "disc_localization")
        mlflow.log_param("dataset",     "REFUGE2")
        mlflow.log_param("num_classes", 1)

        mlflow.log_metric("final_mAP50",     metrics["mAP50"])
        mlflow.log_metric("final_mAP50_95",  metrics["mAP50_95"])
        mlflow.log_metric("final_precision", metrics["precision"])
        mlflow.log_metric("final_recall",    metrics["recall"])

        if os.path.exists(best_pt):
            mlflow.log_artifact(best_pt, artifact_path="weights")
            print(f"  ✓ best.pt logged to MLflow artifacts")

        # Log training plots
        plots_dir = os.path.join(runs_dir, "disc_localizer")
        for plot_file in ["results.png", "confusion_matrix.png",
                          "val_batch0_pred.jpg"]:
            plot_path = os.path.join(plots_dir, plot_file)
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

        run_id = mlflow.active_run().info.run_id
        print(f"  ✓ Final run logged: {run_id}")

    # Save to working directory for download
    os.makedirs(MODEL_DIR, exist_ok=True)
    dst = os.path.join(MODEL_DIR, "yolov8n_disc_localizer.pt")
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, dst)
        print(f"  ✓ Model saved to {dst}")

    return dst


# ---------------------------------------------------------------------------
# Step 5 — Visual verification
# ---------------------------------------------------------------------------

def verify_predictions(model, base, n=6):
    import matplotlib.pyplot as plt

    val_img_dir  = os.path.join(base, "val", "images")
    val_mask_dir = os.path.join(base, "val", "mask")
    val_imgs     = sorted(os.listdir(val_img_dir))[:n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes      = axes.ravel()

    for idx, img_file in enumerate(val_imgs):
        stem      = os.path.splitext(img_file)[0]
        img_path  = os.path.join(val_img_dir, img_file)
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # YOLO prediction
        results = model(img_path, verbose=False)[0]
        overlay = image_rgb.copy()

        if len(results.boxes) > 0:
            best_idx    = results.boxes.conf.argmax()
            box         = results.boxes.xyxy[best_idx].cpu().numpy()
            conf        = float(results.boxes.conf[best_idx])
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(overlay, f"conf={conf:.2f}", (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # GT bbox from mask
        for ext in [".bmp", ".png", ".jpg"]:
            mask_path = os.path.join(val_mask_dir, stem + ext)
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_disc = (gt_mask <= 128).astype(np.uint8)
                ys, xs  = np.where(gt_disc > 0)
                if len(ys) > 0:
                    cv2.rectangle(overlay,
                                  (xs.min(), ys.min()),
                                  (xs.max(), ys.max()),
                                  (255, 255, 0), 2)
                break

        axes[idx].imshow(overlay)
        axes[idx].set_title(
            f"{img_file}\ngreen=pred  yellow=GT", fontsize=9
        )
        axes[idx].axis("off")

    plt.suptitle("YOLOv8n Disc Localization — Validation Samples",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_dir = os.getenv("RUNS_DIR", "/kaggle/working/yolo_runs")
    plt.savefig(os.path.join(output_dir, "yolo_verification.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("✓ Verification plot saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Step 1 — Generate dataset
total = generate_yolo_dataset(BASE, YOLO_DIR)
if total == 0:
    raise ValueError("No labels generated — check BASE path")

yaml_path = write_yolo_yaml(YOLO_DIR)

# Step 2 — Train (streams live to Dagshub)
model, train_results = train_yolo(yaml_path, RUNS_DIR)

# Step 3 — Evaluate
metrics = evaluate_model(model, yaml_path)

# Step 4 — Log final artifacts
model_path = log_final_to_mlflow(model, metrics, RUNS_DIR)

# Step 5 — Verify visually
verify_predictions(model, BASE, n=6)

print(f"\n{'='*50}")
print(f"✓ YOLO training complete")
print(f"  mAP@50:    {metrics['mAP50']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  Model:     {model_path}")
print(f"  Dagshub:   https://dagshub.com/jani2904-sys/NAION-Risk-Analyzer.mlflow")
print(f"{'='*50}")
