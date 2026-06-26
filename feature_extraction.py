import mlflow
import mlflow.pytorch
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import mlflow.pytorch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.measure import label
from ultralytics import YOLO
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = 384
DISC_THRESHOLD   = 0.7918
CUP_THRESHOLD    = 0.5775
CROP_PADDING     = 0.25
BASE             = os.getenv("REFUGE2_BASE",    "/kaggle/input/datasets/victorlemosml/refuge2/REFUGE2")
YOLO_PATH        = os.getenv("YOLO_MODEL_PATH", "weights/yolov8n_disc_localizer.pt")
OUTPUT_DIR       = os.getenv("OUTPUT_DIR",      "naion_results")

MIN_DISC_PX = 5000
MAX_DISC_PX = 150000

# ---------------------------------------------------------------------------
# Dagshub + MLflow
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_segmentation_model():
    model = mlflow.pytorch.load_model("models:/UNet_EfficientNetB4_OpticDisc/latest")
    model = model.to(DEVICE)
    model.eval()
    return model


def load_yolo_model(yolo_path):
    return YOLO(yolo_path)


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------
def get_largest_component(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    counts        = np.bincount(labeled.ravel())
    largest_label = np.argmax(counts[1:]) + 1
    return (labeled == largest_label).astype(np.uint8)


def yolo_locate_disc(image_rgb, yolo_model):
    orig_h, orig_w = image_rgb.shape[:2]
    image_bgr      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results        = yolo_model(image_bgr, verbose=False)[0]

    if len(results.boxes) == 0:
        print("  WARNING: YOLO found no disc — using image center fallback")
        cx, cy = orig_w // 2, orig_h // 2
        half   = min(orig_w, orig_h) // 3
        return max(0, cx-half), max(0, cy-half), \
               min(orig_w, cx+half), min(orig_h, cy+half), None

    best_idx        = results.boxes.conf.argmax()
    box             = results.boxes.xyxy[best_idx].cpu().numpy()
    conf            = float(results.boxes.conf[best_idx])
    x1, y1, x2, y2 = map(int, box)

    pad_x = int((x2 - x1) * CROP_PADDING)
    pad_y = int((y2 - y1) * CROP_PADDING)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(orig_w, x2 + pad_x)
    y2 = min(orig_h, y2 + pad_y)

    return x1, y1, x2, y2, conf


def predict_masks(image_rgb, seg_model, yolo_model, transform):
    orig_h, orig_w        = image_rgb.shape[:2]
    x1, y1, x2, y2, conf = yolo_locate_disc(image_rgb, yolo_model)

    crop           = image_rgb[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    tensor_crop = transform(image=crop)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs_crop = torch.sigmoid(seg_model(tensor_crop)).squeeze(0)

    disc_probs_crop = cv2.resize(
        probs_crop[0].cpu().numpy(), (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR
    )
    cup_probs_crop = cv2.resize(
        probs_crop[1].cpu().numpy(), (crop_w, crop_h),
        interpolation=cv2.INTER_LINEAR
    )

    disc_mask_crop = get_largest_component(
        (disc_probs_crop > DISC_THRESHOLD).astype(np.uint8)
    )
    cup_mask_crop = get_largest_component(
        (cup_probs_crop > CUP_THRESHOLD).astype(np.uint8)
    )
    cup_mask_crop = np.logical_and(cup_mask_crop, disc_mask_crop).astype(np.uint8)

    disc_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cup_mask_full  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    disc_mask_full[y1:y2, x1:x2] = disc_mask_crop
    cup_mask_full[y1:y2,  x1:x2] = cup_mask_crop

    return disc_mask_full, cup_mask_full, conf


# ---------------------------------------------------------------------------
# Base feature helpers
# ---------------------------------------------------------------------------
def get_vertical_diameter(mask):
    ys = np.where(mask > 0)[0]
    return int(ys.max() - ys.min()) if len(ys) > 0 else 0


def get_horizontal_diameter(mask):
    xs = np.where(mask > 0)[1]
    return int(xs.max() - xs.min()) if len(xs) > 0 else 0


def get_mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, None
    return int(np.mean(ys)), int(np.mean(xs))


def get_rim_thickness(disc_mask, cup_mask):
    disc_dist        = cv2.distanceTransform(
        disc_mask.astype(np.uint8), cv2.DIST_L2, 5
    )
    cup_boundary     = cv2.Canny(cup_mask.astype(np.uint8) * 255, 100, 200)
    cup_boundary_pts = np.where(cup_boundary > 0)
    if len(cup_boundary_pts[0]) == 0:
        return 0.0
    return float(disc_dist[cup_boundary_pts].min())


# ---------------------------------------------------------------------------
# Feature 1 — Inferior Rim Thinning (ISNT Rule)
# ---------------------------------------------------------------------------
# Divides the neuroretinal rim into 4 sectors around the disc centroid.
# NAION preferentially damages the inferior rim first.
# inferior_rim_ratio = inferior sector thickness / mean sector thickness
# < 0.85 = genuinely thin (based on dataset mean=0.961, std=0.043)
# ---------------------------------------------------------------------------
def compute_inferior_rim_thinning(disc_mask, cup_mask):
    empty = {
        "inferior_rim_ratio": 1.0,
        "isnt_violation":     False,
        "rim_inferior_px":    0.0,
        "rim_superior_px":    0.0,
        "rim_nasal_px":       0.0,
        "rim_temporal_px":    0.0,
    }

    if disc_mask.sum() == 0 or cup_mask.sum() == 0:
        return empty

    rim_mask  = np.logical_and(
        disc_mask, np.logical_not(cup_mask)
    ).astype(np.uint8)
    disc_dist = cv2.distanceTransform(disc_mask.astype(np.uint8), cv2.DIST_L2, 5)

    disc_cy, disc_cx = get_mask_centroid(disc_mask)
    if disc_cy is None:
        return empty

    ys, xs = np.where(rim_mask > 0)

    def mean_thickness(pts_mask):
        pts_y = ys[pts_mask]
        pts_x = xs[pts_mask]
        if len(pts_y) == 0:
            return 0.0
        return float(disc_dist[pts_y, pts_x].mean())

    rim_superior = mean_thickness(ys < disc_cy)
    rim_inferior = mean_thickness(ys > disc_cy)
    rim_nasal    = mean_thickness(xs < disc_cx)
    rim_temporal = mean_thickness(xs > disc_cx)

    mean_rim = max(
        np.mean([rim_superior, rim_inferior, rim_nasal, rim_temporal]),
        1e-6
    )
    inferior_rim_ratio = round(rim_inferior / mean_rim, 4)
    isnt_violation     = rim_inferior < max(rim_superior, rim_nasal, rim_temporal)

    return {
        "inferior_rim_ratio": inferior_rim_ratio,
        "isnt_violation":     bool(isnt_violation),
        "rim_inferior_px":    round(rim_inferior, 2),
        "rim_superior_px":    round(rim_superior, 2),
        "rim_nasal_px":       round(rim_nasal, 2),
        "rim_temporal_px":    round(rim_temporal, 2),
    }


# ---------------------------------------------------------------------------
# Feature 2 — Disc-Fovea Distance
# ---------------------------------------------------------------------------
# Computed and stored in CSV for reference but NOT used in risk scoring
# because the heuristic is not reliable across different camera FOVs.
# (All 400 images scored below threshold, indicating systematic bias.)
# Will be re-enabled once validated on known cases or replaced by a
# dedicated fovea localization model.
# ---------------------------------------------------------------------------
def compute_disc_fovea_distance(disc_mask, image_rgb):
    empty = {
        "disc_fovea_dist_px": 0.0,
        "disc_fovea_ratio":   0.0,
        "fovea_x":            0,
        "fovea_y":            0,
    }

    if disc_mask.sum() == 0:
        return empty

    disc_cy, disc_cx = get_mask_centroid(disc_mask)
    disc_vd = get_vertical_diameter(disc_mask)
    disc_hd = get_horizontal_diameter(disc_mask)
    disc_d  = (disc_vd + disc_hd) / 2.0

    if disc_d == 0:
        return empty

    h, w      = disc_mask.shape
    gray      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (31, 31), 0)

    best_fovea_x  = disc_cx
    best_fovea_y  = disc_cy
    best_darkness = float('inf')

    for direction in [-1, 1]:
        est_x = int(disc_cx + direction * 2.5 * disc_d)
        est_y = int(disc_cy + 0.3 * disc_d)
        x1    = max(0, est_x - int(disc_d * 0.8))
        x2    = min(w, est_x + int(disc_d * 0.8))
        y1    = max(0, est_y - int(disc_d * 0.8))
        y2    = min(h, est_y + int(disc_d * 0.8))

        if x2 <= x1 or y2 <= y1:
            continue

        region  = gray_blur[y1:y2, x1:x2]
        min_val = region.min()
        min_pos = np.unravel_index(region.argmin(), region.shape)
        fovea_y = y1 + min_pos[0]
        fovea_x = x1 + min_pos[1]
        dist    = np.sqrt((fovea_x - disc_cx)**2 + (fovea_y - disc_cy)**2)

        if 1.5 * disc_d < dist < 5.0 * disc_d and min_val < best_darkness:
            best_darkness = min_val
            best_fovea_x  = fovea_x
            best_fovea_y  = fovea_y

    dist_px    = float(np.sqrt(
        (best_fovea_x - disc_cx)**2 + (best_fovea_y - disc_cy)**2
    ))
    dist_ratio = round(dist_px / disc_d, 4) if disc_d > 0 else 0.0

    return {
        "disc_fovea_dist_px": round(dist_px, 2),
        "disc_fovea_ratio":   dist_ratio,
        "fovea_x":            int(best_fovea_x),
        "fovea_y":            int(best_fovea_y),
    }


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------
def extract_features(disc_mask, cup_mask, image_rgb):
    if disc_mask.sum() == 0:
        return None

    disc_area = int(disc_mask.sum())
    cup_area  = int(cup_mask.sum())
    rim_area  = disc_area - cup_area

    disc_vd = get_vertical_diameter(disc_mask)
    disc_hd = get_horizontal_diameter(disc_mask)
    cup_vd  = get_vertical_diameter(cup_mask)
    cup_hd  = get_horizontal_diameter(cup_mask)

    vCDR     = round(cup_vd / disc_vd, 4) if disc_vd > 0 else 0.0
    hCDR     = round(cup_hd / disc_hd, 4) if disc_hd > 0 else 0.0
    area_CDR = round(cup_area / disc_area, 4) if disc_area > 0 else 0.0

    rim_thickness_px    = get_rim_thickness(disc_mask, cup_mask)
    rim_thickness_ratio = round(rim_thickness_px / disc_vd, 4) if disc_vd > 0 else 0.0

    disc_diameter_avg = (disc_vd + disc_hd) / 2.0
    CDI = round(
        disc_area / (disc_diameter_avg ** 2), 4
    ) if disc_diameter_avg > 0 else 0.0

    disc_roundness = round(
        min(disc_vd, disc_hd) / max(disc_vd, disc_hd), 4
    ) if max(disc_vd, disc_hd) > 0 else 0.0

    isnt_features  = compute_inferior_rim_thinning(disc_mask, cup_mask)
    fovea_features = compute_disc_fovea_distance(disc_mask, image_rgb)

    features = {
        "disc_area_px":        disc_area,
        "cup_area_px":         cup_area,
        "rim_area_px":         rim_area,
        "disc_vd_px":          disc_vd,
        "disc_hd_px":          disc_hd,
        "cup_vd_px":           cup_vd,
        "cup_hd_px":           cup_hd,
        "vCDR":                vCDR,
        "hCDR":                hCDR,
        "area_CDR":            area_CDR,
        "rim_thickness_px":    round(rim_thickness_px, 2),
        "rim_thickness_ratio": rim_thickness_ratio,
        "CDI":                 CDI,
        "disc_roundness":      disc_roundness,
    }
    features.update(isnt_features)
    features.update(fovea_features)
    return features


def is_valid(features):
    if features is None:
        return False, "disc mask empty"
    if features["vCDR"] == 0.0 or features["cup_area_px"] < 100:
        return False, "cup not detected"
    if features["disc_area_px"] > MAX_DISC_PX:
        return False, f"disc too large ({features['disc_area_px']}px)"
    if features["disc_area_px"] < MIN_DISC_PX:
        return False, f"disc too small ({features['disc_area_px']}px)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Rule-Based NAION Risk Score
# ---------------------------------------------------------------------------
# Rules and weights recalibrated based on actual data distributions:
#   vCDR:               mean=0.505, range 0.38-0.82
#   CDI:                mean=0.767, range 0.68-0.83
#   rim_thickness_ratio: mean=0.108, range 0.03-0.24
#   inferior_rim_ratio:  mean=0.961, std=0.043
#
# disc_fovea_ratio EXCLUDED from scoring — heuristic unreliable across FOVs
# (all 400 images scored below threshold → systematic bias confirmed)
# ---------------------------------------------------------------------------
def compute_naion_risk_score(features):
    if features is None:
        return 0, "Unknown", {}

    points     = 0
    max_points = 0
    breakdown  = {}

    valid, reason = is_valid(features)
    if not valid:
        return 0, "Invalid", {f"Segmentation failed: {reason}": 0}

    # --- Rule 1: vCDR (35 points) ---
    # Primary NAION risk factor — small cup = crowded disc
    max_points += 35
    vCDR = features["vCDR"]
    if vCDR < 0.2:
        p = 35; breakdown["vCDR < 0.2 (severely crowded)"] = p
    elif vCDR < 0.3:
        p = 30; breakdown["vCDR < 0.3 (crowded)"] = p
    elif vCDR < 0.4:
        p = 20; breakdown["vCDR < 0.4 (moderately crowded)"] = p
    elif vCDR < 0.5:
        p = 10; breakdown["vCDR < 0.5 (mildly crowded)"] = p
    else:
        p = 0;  breakdown["vCDR >= 0.5 (normal/glaucomatous)"] = p
    points += p

    # --- Rule 2: CDI (20 points) ---
    # Recalibrated: dataset range 0.68-0.83, mean=0.767
    max_points += 20
    CDI = features["CDI"]
    if CDI > 0.82:
        p = 20; breakdown["CDI > 0.82 (crowded disc)"] = p
    elif CDI > 0.78:
        p = 12; breakdown["CDI > 0.78 (borderline crowded)"] = p
    elif CDI > 0.75:
        p = 6;  breakdown["CDI > 0.75 (mildly crowded)"] = p
    else:
        p = 0;  breakdown["CDI <= 0.75 (normal disc size)"] = p
    points += p

    # --- Rule 3: Rim thickness ratio (20 points) ---
    # Recalibrated: dataset mean=0.108, range 0.03-0.24
    max_points += 20
    rim_ratio = features["rim_thickness_ratio"]
    if rim_ratio < 0.05:
        p = 20; breakdown["Rim ratio < 0.05 (very thin rim)"] = p
    elif rim_ratio < 0.08:
        p = 14; breakdown["Rim ratio < 0.08 (thin rim)"] = p
    elif rim_ratio < 0.12:
        p = 7;  breakdown["Rim ratio < 0.12 (borderline rim)"] = p
    else:
        p = 0;  breakdown["Rim ratio >= 0.12 (normal rim)"] = p
    points += p

    # --- Rule 4: Area CDR (15 points) ---
    max_points += 15
    area_CDR = features["area_CDR"]
    if area_CDR < 0.10:
        p = 15; breakdown["Area CDR < 0.10 (very small cup)"] = p
    elif area_CDR < 0.20:
        p = 10; breakdown["Area CDR < 0.20 (small cup)"] = p
    elif area_CDR < 0.30:
        p = 5;  breakdown["Area CDR < 0.30 (borderline)"] = p
    else:
        p = 0;  breakdown["Area CDR >= 0.30 (normal)"] = p
    points += p

    # --- Rule 5: Disc roundness (5 points) ---
    # Reduced weight — not highly discriminating (mean=0.926)
    max_points += 5
    roundness = features["disc_roundness"]
    if roundness > 0.97:
        p = 5;  breakdown["Disc roundness > 0.97 (extremely round)"] = p
    elif roundness > 0.93:
        p = 2;  breakdown["Disc roundness > 0.93 (very round)"] = p
    else:
        p = 0;  breakdown["Disc roundness <= 0.93 (normal shape)"] = p
    points += p

    # --- Rule 6: Inferior rim thinning (20 points) ---
    # Recalibrated: dataset mean=0.961, std=0.043
    # < 0.85 = mean - 2.5 std = genuinely abnormal thinning
    max_points += 20
    inf_ratio = features["inferior_rim_ratio"]
    if inf_ratio < 0.85:
        p = 20; breakdown["Inferior rim ratio < 0.85 (severe inferior thinning)"] = p
    elif inf_ratio < 0.90:
        p = 13; breakdown["Inferior rim ratio < 0.90 (inferior thinning)"] = p
    elif inf_ratio < 0.93:
        p = 6;  breakdown["Inferior rim ratio < 0.93 (mild inferior thinning)"] = p
    else:
        p = 0;  breakdown["Inferior rim ratio >= 0.93 (normal inferior rim)"] = p
    points += p

    # ISNT violation bonus (5 points)
    if features["isnt_violation"]:
        max_points += 5
        points     += 5
        breakdown["ISNT rule violated (inferior not thickest sector)"] = 5

    # Note: disc_fovea_ratio is computed and saved in CSV but excluded
    # from scoring — heuristic unreliable across different camera FOVs

    score      = int(round((points / max_points) * 100))
    risk_level = "High" if score >= 65 else "Moderate" if score >= 40 else "Low"

    return score, risk_level, breakdown


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_result_card(image_rgb, disc_mask, cup_mask, features, score,
                     risk_level, breakdown, image_name, yolo_conf, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    overlay   = image_rgb.copy()
    disc_c, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_c,  _ = cv2.findContours(cup_mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, disc_c, -1, (0, 255, 0), 3)
    cv2.drawContours(overlay, cup_c,  -1, (255, 0, 0), 3)

    fovea_x = features.get("fovea_x", 0)
    fovea_y = features.get("fovea_y", 0)
    if fovea_x > 0 and fovea_y > 0:
        cv2.circle(overlay, (fovea_x, fovea_y), 15, (255, 165, 0), 3)

    conf_str = f"YOLO conf={yolo_conf:.2f}" if yolo_conf else "YOLO: fallback"
    axes[0].imshow(overlay)
    axes[0].set_title(f"{image_name}  |  {conf_str}", fontsize=11)
    axes[0].axis("off")
    axes[0].legend(
        handles=[
            mpatches.Patch(color="green",  label="Disc"),
            mpatches.Patch(color="red",    label="Cup"),
            mpatches.Patch(color="orange", label="Fovea estimate (ref only)"),
        ],
        loc="lower right", fontsize=8
    )

    axes[1].axis("off")
    score_color = {"Low": "green", "Moderate": "orange", "High": "red"}[risk_level]

    axes[1].text(0.5, 0.98, f"NAION Risk Score: {score}/100",
                 ha="center", va="top", fontsize=16, fontweight="bold",
                 color=score_color, transform=axes[1].transAxes)
    axes[1].text(0.5, 0.91, f"Risk Level: {risk_level}",
                 ha="center", va="top", fontsize=13, color=score_color,
                 transform=axes[1].transAxes)

    feature_lines = [
        ("vCDR",               f"{features['vCDR']:.3f}",               "< 0.5 = crowded"),
        ("hCDR",               f"{features['hCDR']:.3f}",               "< 0.5 = crowded"),
        ("Area CDR",           f"{features['area_CDR']:.3f}",           "< 0.2 = small cup"),
        ("CDI",                f"{features['CDI']:.3f}",                "> 0.78 = crowded"),
        ("Rim ratio",          f"{features['rim_thickness_ratio']:.3f}","< 0.12 = thin"),
        ("Disc roundness",     f"{features['disc_roundness']:.3f}",     "> 0.93 = round"),
        ("Inferior rim ratio", f"{features['inferior_rim_ratio']:.3f}", "< 0.90 = thinning"),
        ("ISNT violation",     str(features['isnt_violation']),          "True = risk"),
        ("Disc-fovea ratio",   f"{features['disc_fovea_ratio']:.3f}",   "ref only"),
        ("Disc area (px)",     f"{features['disc_area_px']}",           ""),
        ("Cup area (px)",      f"{features['cup_area_px']}",            ""),
    ]

    y = 0.84
    axes[1].text(0.03, y, "Feature",   fontsize=9, fontweight="bold", transform=axes[1].transAxes)
    axes[1].text(0.45, y, "Value",     fontsize=9, fontweight="bold", transform=axes[1].transAxes)
    axes[1].text(0.62, y, "Threshold", fontsize=9, fontweight="bold", transform=axes[1].transAxes)
    y -= 0.025
    axes[1].axhline(y=y, xmin=0.02, xmax=0.98, color="gray", linewidth=0.5)

    for fname, fval, fthresh in feature_lines:
        y -= 0.052
        axes[1].text(0.03, y, fname,   fontsize=8, transform=axes[1].transAxes)
        axes[1].text(0.45, y, fval,    fontsize=8, transform=axes[1].transAxes)
        axes[1].text(0.62, y, fthresh, fontsize=8, color="gray", transform=axes[1].transAxes)

    y -= 0.06
    axes[1].text(0.03, y, "Score breakdown:", fontsize=9,
                 fontweight="bold", transform=axes[1].transAxes)
    for rule, pts in breakdown.items():
        if pts > 0:
            y -= 0.046
            axes[1].text(0.03, y, f"  +{pts}  {rule}", fontsize=7.5,
                         color="darkred", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    seg_model  = load_segmentation_model()
    yolo_model = load_yolo_model(YOLO_PATH)
    transform  = A.Compose([
        A.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])

    test_pairs    = get_pairs(BASE, "test")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results       = []
    valid_count   = 0
    invalid_count = 0

    for i, (img_path, mask_path) in enumerate(tqdm(test_pairs)):
        image_name = os.path.basename(img_path)
        image_bgr  = cv2.imread(img_path)
        image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        disc_mask, cup_mask, yolo_conf = predict_masks(
            image_rgb, seg_model, yolo_model, transform
        )
        features = extract_features(disc_mask, cup_mask, image_rgb)

        valid, reason = is_valid(features)
        if not valid:
            print(f"{image_name} | INVALID ({reason}) — skipped")
            invalid_count += 1
            continue

        score, risk_level, breakdown = compute_naion_risk_score(features)
        valid_count += 1

        conf_str = f"{yolo_conf:.2f}" if yolo_conf else "fallback"
        print(
            f"{image_name} | "
            f"vCDR={features['vCDR']:.3f} | "
            f"inf_rim={features['inferior_rim_ratio']:.3f} | "
            f"Score={score} | Risk={risk_level} | YOLO={conf_str}"
        )

        if risk_level != "Invalid":
            save_path = os.path.join(OUTPUT_DIR, f"result_{i:03d}_{image_name}.png")
            save_result_card(
                image_rgb=image_rgb,
                disc_mask=disc_mask,
                cup_mask=cup_mask,
                features=features,
                score=score,
                risk_level=risk_level,
                breakdown=breakdown,
                image_name=image_name,
                yolo_conf=yolo_conf,
                save_path=save_path
            )

        row = {
            "image":      image_name,
            "risk_score": score,
            "risk_level": risk_level,
            "yolo_conf":  yolo_conf,
        }
        row.update(features)
        results.append(row)

    df       = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "naion_risk_scores.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✓ Processed {valid_count} valid / {invalid_count} invalid images")
    print(f"✓ CSV saved to {csv_path}")

    print(f"\n--- Risk Level Distribution ---")
    print(df["risk_level"].value_counts().to_string())

    print(f"\n--- Feature Summary ---")
    summary_cols = [
        "vCDR", "CDI", "rim_thickness_ratio",
        "inferior_rim_ratio", "disc_fovea_ratio", "risk_score"
    ]
    print(df[summary_cols].describe().round(3).to_string())

    print(f"\n--- NAION Suspicious (vCDR < 0.4) ---")
    suspicious = df[df["vCDR"] < 0.4]
    print(suspicious[[
        "image", "vCDR", "CDI", "rim_thickness_ratio",
        "inferior_rim_ratio", "isnt_violation",
        "disc_fovea_ratio", "risk_score", "risk_level"
    ]].sort_values("vCDR").to_string())


if __name__ == "__main__":
    main()

