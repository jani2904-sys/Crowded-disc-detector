import os
import glob
import cv2
import numpy as np
import kagglehub
import torch
from torch.utils.data import Dataset

# 1. Automate data acquisition
DEFAULT_BASE = os.getenv("REFUGE2_BASE", "/kaggle/input/refuge2/REFUGE2")


def _resolve_base_path():
    def has_split_dirs(path):
        if not os.path.isdir(path):
            return False
        split_names = ["train", "val", "validation", "test"]
        return any(os.path.isdir(os.path.join(path, s)) for s in split_names)

    env_base = os.getenv("REFUGE2_BASE")
    if env_base and os.path.isdir(env_base):
        return env_base

    if os.path.isdir(DEFAULT_BASE):
        return DEFAULT_BASE

    # Fallback for non-Kaggle environments: download to local cache.
    try:
        downloaded_root = kagglehub.dataset_download("victorlemosml/refuge2")
        candidates = [
            os.path.join(downloaded_root, "REFUGE2"),
            downloaded_root,
        ]
        for cand in candidates:
            if has_split_dirs(cand):
                return cand
        for cand in candidates:
            if os.path.isdir(cand):
                return cand
    except Exception as exc:
        print(f"DEBUG: kagglehub download unavailable: {exc}")

    # Keep deterministic output for callers even when data is unavailable.
    return env_base or DEFAULT_BASE


BASE = _resolve_base_path()
print(f"Dataset base resolved to: {BASE}")

# 2. Add your get_pairs function


def _first_existing_dir(parent, candidates):
    for name in candidates:
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            return path
    return None


def _resolve_split_dir(base, split):
    candidates = [split, split.lower(), split.upper(), split.capitalize()]
    return _first_existing_dir(base, candidates)

def get_pairs(base, split):
    # Support 'test' split specifically for the 400 official test images
    split_dir = _resolve_split_dir(base, split)
    pairs = []

    if split_dir is None:
        print(f"DEBUG: No split directory found for '{split}' under {base}.")
        return []

    img_dir = _first_existing_dir(split_dir, ["images", "image", "Images", "Image"])
    mask_dir = _first_existing_dir(split_dir, ["mask", "masks", "Mask", "Masks"])

    if img_dir is None:
        print(f"DEBUG: No images directory found under {split_dir}.")
        return []

    # Handle cases where test set might not have masks (blind test)
    if mask_dir is None:
        print(f"DEBUG: No masks found for split '{split}' under {split_dir}, proceeding with images only.")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        return [(p, None) for p in img_paths]

    # ... (existing pair-matching logic for train/val) ...
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0]
        img_path = os.path.join(img_dir, f)
        mask_path = None
        for ext in [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            cand = os.path.join(mask_dir, stem + ext)
            if os.path.exists(cand):
                mask_path = cand
                break
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs
    
# 3. Add your Refuge2Dataset class
class Refuge2Dataset(Dataset):
    def __init__(self, pairs, transform=None, crop_size=640):
        self.pairs = pairs
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Test-time: if no mask, skip cropping logic and return just image
        if mask_path is None:
            if self.transform:
                image = self.transform(image=image)['image']
            if isinstance(image, torch.Tensor):
                h, w = image.shape[-2], image.shape[-1]
            else:
                h, w = image.shape[0], image.shape[1]
            return image, torch.zeros((2, h, w), dtype=torch.float32)
            
        full_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if full_mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        
        ys, xs = np.where(full_mask <= 128)
        if len(ys) > 0:
            center_y, center_x = int(np.mean(ys)), int(np.mean(xs))
            half = self.crop_size // 2
            y1, y2 = max(0, center_y - half), min(image.shape[0], center_y + half)
            x1, x2 = max(0, center_x - half), min(image.shape[1], center_x + half)
            image, full_mask = image[y1:y2, x1:x2], full_mask[y1:y2, x1:x2]
        
        mask = np.stack([(full_mask <= 128).astype(float), (full_mask == 0).astype(float)], axis=-1)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask'].permute(2, 0, 1)
        return image, mask
