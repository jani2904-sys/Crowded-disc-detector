import os
import cv2
import numpy as np
import torch
import mlflow.pytorch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from ultralytics import YOLO
from skimage.measure import label

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = 384
DISC_THRESHOLD   = 0.7918
CUP_THRESHOLD    = 0.5775
CROP_PADDING     = 0.25    # extra padding around YOLO bbox before cropping

# ---------------------------------------------------------------------------
# Dagshub + MLflow
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_segmentation_model():
    import os
    import torch
    import mlflow.pytorch

    # Force CPU mapping — required for environments without GPU
    # (Streamlit Cloud, CI, local CPU machines)
    os.environ["MLFLOW_DEFAULT_PYTORCH_DEVICE"] = "cpu"

    # Patch torch.load to always use CPU map_location
    original_torch_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)
    torch.load = patched_load

    try:
        model = mlflow.pytorch.load_model(
            "models:/UNet_EfficientNetB4_OpticDisc/latest"
        )
    finally:
        # Restore original torch.load after model is loaded
        torch.load = original_torch_load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()
    return model


def _load_yolo_model(yolo_path):
    """
    Load YOLOv8n disc localizer.
    yolo_path: path to yolov8n_disc_localizer.pt
    """
    model = YOLO(yolo_path)
    return model


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def _build_transform():
    return A.Compose([
        A.Resize(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])


# ---------------------------------------------------------------------------
# Largest connected component
# ---------------------------------------------------------------------------
def _get_largest_component(mask):
    labeled = label(mask)
    if labeled.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    counts        = np.bincount(labeled.ravel())
    largest_label = np.argmax(counts[1:]) + 1
    return (labeled == largest_label).astype(np.uint8)


# ---------------------------------------------------------------------------
# YOLO-based disc localization
# ---------------------------------------------------------------------------
def _yolo_locate_disc(image_rgb, yolo_model):
    """
    Uses YOLOv8n to find the optic disc bounding box.

    Returns (x1, y1, x2, y2) with padding, or None if not detected.

    IMPROVEMENT over heuristic:
      Old: brightness + model probability → failed on bright lesions
           (T0127, T0132, T0134 all had wrong crops)
      New: YOLO trained specifically on disc appearance →
           robust to macular pathology, bright lesions, edge discs
    """
    orig_h, orig_w = image_rgb.shape[:2]

    # YOLO expects BGR
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results   = yolo_model(image_bgr, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    # Take highest confidence detection
    best_idx    = results.boxes.conf.argmax()
    box         = results.boxes.xyxy[best_idx].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    # Add padding around bbox so segmentation model has context
    pad_x = int((x2 - x1) * CROP_PADDING)
    pad_y = int((y2 - y1) * CROP_PADDING)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(orig_w, x2 + pad_x)
    y2 = min(orig_h, y2 + pad_y)

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Main predict function
# ---------------------------------------------------------------------------
def predict(image_input, seg_model, yolo_model):
    """
    Run disc + cup segmentation on a single image.

    Args:
        image_input : file path (str), file-like object, or numpy RGB array
        seg_model   : EfficientNet-b4 UNet loaded via _load_segmentation_model()
        yolo_model  : YOLOv8n loaded via _load_yolo_model()

    Returns dict with:
        disc_mask_full : binary np.uint8 at original resolution
        cup_mask_full  : binary np.uint8 at original resolution
        disc_probs     : float32 probability map at model resolution
        cup_probs      : float32 probability map at model resolution
        crop_coords    : (x1, y1, x2, y2) of the crop used
        original_size  : (width, height) of input image
        yolo_conf      : YOLO detection confidence (None if fallback used)
    """
    transform = _build_transform()

    # --- Load image ---
    if isinstance(image_input, np.ndarray):
        image_rgb = image_input
    elif isinstance(image_input, str):
        image_bgr = cv2.imread(image_input)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        pil_img   = Image.open(image_input).convert("RGB")
        image_rgb = np.array(pil_img)

    orig_h, orig_w = image_rgb.shape[:2]

    # --- Step 1: YOLO disc localization ---
    bbox     = _yolo_locate_disc(image_rgb, yolo_model)
    yolo_conf = None

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Get confidence for reporting
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results   = yolo_model(image_bgr, verbose=False)[0]
        if len(results.boxes) > 0:
            yolo_conf = float(results.boxes.conf.max())
    else:
        # Fallback — use image center if YOLO finds nothing
        print("WARNING: YOLO found no disc — using image center as fallback")
        cx, cy = orig_w // 2, orig_h // 2
        half   = min(orig_w, orig_h) // 3
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(orig_w, cx + half)
        y2 = min(orig_h, cy + half)

    # --- Step 2: Crop ROI ---
    crop           = image_rgb[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    # --- Step 3: Segmentation on crop ---
    tensor_crop = transform(image=crop)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs_crop = torch.sigmoid(seg_model(tensor_crop)).squeeze(0)

    disc_probs_np = probs_crop[0].cpu().numpy()
    cup_probs_np  = probs_crop[1].cpu().numpy()

    disc_probs_crop = cv2.resize(disc_probs_np, (crop_w, crop_h),
                                 interpolation=cv2.INTER_LINEAR)
    cup_probs_crop  = cv2.resize(cup_probs_np,  (crop_w, crop_h),
                                 interpolation=cv2.INTER_LINEAR)

    disc_mask_crop = (disc_probs_crop > DISC_THRESHOLD).astype(np.uint8)
    cup_mask_crop  = (cup_probs_crop  > CUP_THRESHOLD).astype(np.uint8)

    disc_mask_crop = _get_largest_component(disc_mask_crop)
    cup_mask_crop  = _get_largest_component(cup_mask_crop)
    cup_mask_crop  = np.logical_and(cup_mask_crop, disc_mask_crop).astype(np.uint8)

    # --- Step 4: Paste back into full image canvas ---
    disc_mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cup_mask_full  = np.zeros((orig_h, orig_w), dtype=np.uint8)
    disc_mask_full[y1:y2, x1:x2] = disc_mask_crop
    cup_mask_full[y1:y2,  x1:x2] = cup_mask_crop

    return {
        "disc_mask_full": disc_mask_full,
        "cup_mask_full":  cup_mask_full,
        "disc_probs":     disc_probs_np,
        "cup_probs":      cup_probs_np,
        "crop_coords":    (x1, y1, x2, y2),
        "original_size":  (orig_w, orig_h),
        "yolo_conf":      yolo_conf,
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    YOLO_PATH = os.getenv("YOLO_MODEL_PATH", "weights/yolov8n_disc_localizer.pt")

    print(f"Device: {DEVICE}")
    print(f"Loading segmentation model from Dagshub...")
    seg_model  = _load_segmentation_model()

    print(f"Loading YOLO model from {YOLO_PATH}...")
    yolo_model = _load_yolo_model(YOLO_PATH)

    print(f"Running test on T0001.jpg...")
    BASE      = os.getenv("REFUGE2_BASE", "/kaggle/input/datasets/victorlemosml/refuge2/REFUGE2")
    test_path = f"{BASE}/test/images/T0001.jpg"
    result    = predict(test_path, seg_model, yolo_model)

    print(f"Disc mask shape:   {result['disc_mask_full'].shape}")
    print(f"Cup mask shape:    {result['cup_mask_full'].shape}")
    print(f"Disc pixels:       {result['disc_mask_full'].sum()}")
    print(f"Cup pixels:        {result['cup_mask_full'].sum()}")
    print(f"Crop coords:       {result['crop_coords']}")
    print(f"YOLO confidence:   {result['yolo_conf']:.3f}" if result['yolo_conf'] else "YOLO: fallback used")
    print("✓ Inference working correctly")
