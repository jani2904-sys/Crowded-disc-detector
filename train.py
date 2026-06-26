import torch
import torch.nn as nn
import os
import sys
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import threading
import time
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Dagshub + MLflow Tracking
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE      = 6
LEARNING_RATE   = 0.0003
EPOCHS          = 30
CHECKPOINT_DIR  = "/kaggle/working/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ---------------------------------------------------------------------------
# Keep-alive
# ---------------------------------------------------------------------------
KEEP_ALIVE_LOG        = "training_progress.log"
keep_alive_stop_event = threading.Event()

def keep_alive_worker():
    counter = 0
    while not keep_alive_stop_event.is_set():
        time.sleep(30)
        counter += 1
        try:
            with open(KEEP_ALIVE_LOG, 'a') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Keep-alive {counter}...\n")
                f.flush()
            print(f"[Keep-alive] Training in progress... ({counter * 30}s elapsed)", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"Keep-alive error: {e}")

# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        probs   = probs.view(probs.shape[0], probs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        intersection = (probs * targets).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=2) + targets.sum(dim=2) + self.smooth
        )
        return 1.0 - dice.mean()


class CupWeightedLoss(nn.Module):
    def __init__(self, disc_weight=1.0, cup_weight=3.0):
        super().__init__()
        self.disc_weight = disc_weight
        self.cup_weight  = cup_weight
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        weights  = torch.tensor(
            [self.disc_weight, self.cup_weight],
            dtype=torch.float32,
            device=logits.device
        )
        bce_loss  = self.bce(logits, targets)
        bce_loss  = (bce_loss * weights[None, :, None, None]).mean()
        dice_loss = self.dice(logits, targets)
        return bce_loss + dice_loss

# ---------------------------------------------------------------------------
# Per-channel Dice
# ---------------------------------------------------------------------------
def compute_dice_per_channel(preds, targets, threshold=0.5):
    probs  = torch.sigmoid(preds)
    binary = (probs > threshold).float()
    dice_scores = []
    for c in range(binary.shape[1]):
        b = binary[:, c]
        t = targets[:, c]
        intersection = (b * t).sum()
        dice = (2.0 * intersection) / (b.sum() + t.sum() + 1e-6)
        dice_scores.append(dice.item())
    return dice_scores[0], dice_scores[1]

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------
def train_model_optimized(
    model,
    model_name,
    train_loader,
    val_loader,
    optimizer,
    scheduler=None,
    start_epoch=0,
    best_cup_dice=0.0,
    patience_counter=0
):
    model     = model.to(DEVICE)
    criterion = CupWeightedLoss(disc_weight=1.0, cup_weight=3.0)
    patience  = 6

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("encoder",       "efficientnet-b4")
        mlflow.log_param("decoder_attn",  "scse")
        mlflow.log_param("loss",          "WeightedBCE+Dice")
        mlflow.log_param("cup_weight",    3.0)
        mlflow.log_param("input_size",    384)
        mlflow.log_param("batch_size",    BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)

        for epoch in range(start_epoch, EPOCHS):

            # --- Training ---
            model.train()
            running_loss = 0.0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                    print(
                        f"[Epoch {epoch+1}/{EPOCHS}] "
                        f"Batch {batch_idx+1}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}",
                        flush=True
                    )
                    sys.stdout.flush()
                elif (batch_idx + 1) % 2 == 0:
                    print(
                        f"  └─ Batch {batch_idx+1}/{len(train_loader)}: "
                        f"{loss.item():.4f}",
                        flush=True
                    )

            train_loss = running_loss / len(train_loader)

            # --- Validation ---
            model.eval()
            val_loss          = 0.0
            val_disc_dice_sum = 0.0
            val_cup_dice_sum  = 0.0

            if len(val_loader) > 0:
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(DEVICE), masks.to(DEVICE)
                        outputs = model(images)
                        val_loss += criterion(outputs, masks).item()
                        disc_d, cup_d = compute_dice_per_channel(outputs, masks)
                        val_disc_dice_sum += disc_d
                        val_cup_dice_sum  += cup_d

                val_loss      /= len(val_loader)
                val_disc_dice  = val_disc_dice_sum / len(val_loader)
                val_cup_dice   = val_cup_dice_sum  / len(val_loader)
            else:
                print("WARNING: Validation loader is empty.", flush=True)
                val_disc_dice = 0.0
                val_cup_dice  = 0.0

            # --- MLflow Metrics ---
            mlflow.log_metric("train_loss",    train_loss,    step=epoch)
            mlflow.log_metric("val_loss",      val_loss,      step=epoch)
            mlflow.log_metric("val_disc_dice", val_disc_dice, step=epoch)
            mlflow.log_metric("val_cup_dice",  val_cup_dice,  step=epoch)

            print(
                f"✓ Epoch {epoch+1}/{EPOCHS} Complete — "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Disc Dice: {val_disc_dice:.4f}, Cup Dice: {val_cup_dice:.4f}",
                flush=True
            )
            sys.stdout.flush()

            # --- Scheduler ---
            if scheduler is not None and len(val_loader) > 0:
                try:
                    scheduler.step(val_loss)
                except Exception:
                    pass

            # --- Early Stopping + Best Model Save ---
            if val_cup_dice > best_cup_dice:
                best_cup_dice    = val_cup_dice
                patience_counter = 0
                print(
                    f"[Early Stopping] Cup Dice improved to {val_cup_dice:.4f}. "
                    f"Resetting patience.",
                    flush=True
                )
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                best_checkpoint = {
                    'epoch':                epoch + 1,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss':           train_loss,
                    'val_loss':             val_loss,
                    'val_disc_dice':        val_disc_dice,
                    'val_cup_dice':         val_cup_dice,
                    'best_cup_dice':        best_cup_dice,
                    'patience_counter':     patience_counter,
                }
                if scheduler is not None:
                    try:
                        best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    except Exception:
                        pass
                torch.save(best_checkpoint, BEST_MODEL_PATH)
                print(f"[Best Model] Saved to {BEST_MODEL_PATH}", flush=True)

                # Save model artifact to Dagshub MLflow registry
                # Save model artifact to Dagshub MLflow registry
               
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="best_model",
                    registered_model_name="UNet_EfficientNetB4_OpticDisc",
                    serialization_format="pickle",  # avoids pt2 tracing requirement
                )
                print("[MLflow] Model artifact logged to Dagshub registry", flush=True)

            else:
                patience_counter += 1
                print(
                    f"[Early Stopping] Cup Dice did not improve "
                    f"({val_cup_dice:.4f} vs best {best_cup_dice:.4f}). "
                    f"Patience: {patience_counter}/{patience}",
                    flush=True
                )
                if patience_counter >= patience:
                    print(
                        f"[Early Stopping] Stopping at epoch {epoch+1} — "
                        f"no improvement for {patience} epochs.",
                        flush=True
                    )
                    break

            # --- Rolling Checkpoint ---
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint = {
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':           train_loss,
                'val_loss':             val_loss,
                'val_disc_dice':        val_disc_dice,
                'val_cup_dice':         val_cup_dice,
                'best_cup_dice':        best_cup_dice,
                'patience_counter':     patience_counter,
            }
            if scheduler is not None:
                try:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                except Exception:
                    pass
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"[Checkpoint] Saved at epoch {epoch+1}", flush=True)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
print("Starting training with idle timeout prevention...", flush=True)
keep_alive_thread = threading.Thread(target=keep_alive_worker, daemon=True)
keep_alive_thread.start()

try:
    BASE = "/kaggle/input/datasets/victorlemosml/refuge2/REFUGE2"

    all_pairs = {}
    for split in ["train", "val"]:
        all_pairs[split] = get_pairs(BASE, split)

    train_pairs = all_pairs["train"]
    val_pairs   = all_pairs["val"]

    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}", flush=True)

    if len(train_pairs) == 0:
        raise ValueError("Training pairs list is empty!")

    train_transform = A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ])

    train_loader = DataLoader(
        Refuge2Dataset(train_pairs, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        Refuge2Dataset(val_pairs, transform=val_transform),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        decoder_attention_type="scse",
        decoder_dropout=0.3
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # --- Checkpoint Resume ---
    start_epoch      = 0
    best_cup_dice    = 0.0
    patience_counter = 0
    checkpoint_file  = None

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint_file = CHECKPOINT_PATH
    else:
        ckpts = sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")),
            key=os.path.getmtime
        )
        if ckpts:
            checkpoint_file = ckpts[-1]

    if checkpoint_file:
        print(f"[Resume] Loading checkpoint from {checkpoint_file}...", flush=True)
        ckpt = torch.load(checkpoint_file, map_location=DEVICE)
        model.load_state_dict(ckpt.get('model_state_dict', {}))
        try:
            optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)
        except Exception as e:
            print(f"[Resume] Warning: failed to load optimizer state: {e}", flush=True)

        start_epoch      = int(ckpt.get('epoch', 0))
        best_cup_dice    = float(ckpt.get('best_cup_dice', ckpt.get('val_cup_dice', 0.0)))
        patience_counter = int(ckpt.get('patience_counter', 0))

        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e:
                print(f"[Resume] Warning: failed to load scheduler state: {e}", flush=True)
        elif 'val_loss' in ckpt:
            try:
                scheduler.step(ckpt['val_loss'])
            except Exception:
                pass

        print(
            f"[Resume] Resuming from epoch {start_epoch}, "
            f"best Cup Dice so far: {best_cup_dice:.4f}",
            flush=True
        )

    train_model_optimized(
        model,
        "UNet_EfficientNetB4_v2",
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        start_epoch=start_epoch,
        best_cup_dice=best_cup_dice,
        patience_counter=patience_counter,
    )

    print("\n✓ Training completed successfully!", flush=True)

finally:
    keep_alive_stop_event.set()
    keep_alive_thread.join(timeout=2)
