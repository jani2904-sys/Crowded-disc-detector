import os
import io
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Load env before importing models
load_dotenv()

from inference import predict, _load_segmentation_model, _load_yolo_model
from feature_extraction import extract_features, compute_naion_risk_score

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crowded disc detector",
    description="Automated NAION risk assessment from fundus images",
    version="1.0.0"
)

# Load models once at startup — not per request
YOLO_PATH  = os.getenv("YOLO_MODEL_PATH", "weights/yolov8n_disc_localizer.pt")
seg_model  = _load_segmentation_model()
yolo_model = _load_yolo_model(YOLO_PATH)

print("✓ Models loaded and ready")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "NAION Risk Analyzer",
        "version": "1.0.0",
        "status":  "running",
        "endpoints": {
            "POST /analyze": "Upload a fundus image and get NAION risk score",
            "GET  /health":  "Health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_fundus(file: UploadFile = File(...)):
    """
    Upload a fundus image (JPEG or PNG) and receive a NAION risk assessment.

    Returns:
        risk_score  : int 0-100
        risk_level  : "Low" / "Moderate" / "High" / "Invalid"
        vCDR        : vertical cup-to-disc ratio
        CDI         : crowded disc index
        rim_ratio   : rim thickness ratio
        inf_rim     : inferior rim ratio
        isnt        : ISNT rule violation (True/False)
        breakdown   : which rules fired and their point values
        yolo_conf   : YOLO detection confidence (null if fallback used)
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only JPEG and PNG are supported."
        )

    try:
        # Read uploaded bytes and convert to numpy RGB array
        image_bytes = await file.read()
        pil_img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb   = np.array(pil_img)

        # Run inference — YOLO localization + segmentation
        result = predict(image_rgb, seg_model, yolo_model)

        # Extract clinical features
        features = extract_features(
            result["disc_mask_full"],
            result["cup_mask_full"],
            image_rgb
        )

        # Compute risk score
        score, risk_level, breakdown = compute_naion_risk_score(features)

        # Build response
        if features is None or risk_level == "Invalid":
            return JSONResponse(
                status_code=200,
                content={
                    "risk_score":  0,
                    "risk_level":  "Invalid",
                    "message":     "Segmentation failed — disc or cup not detected",
                    "yolo_conf":   result.get("yolo_conf"),
                }
            )

        return {
            "risk_score":  score,
            "risk_level":  risk_level,
            "vCDR":        features["vCDR"],
            "hCDR":        features["hCDR"],
            "CDI":         features["CDI"],
            "area_CDR":    features["area_CDR"],
            "rim_ratio":   features["rim_thickness_ratio"],
            "inf_rim":     features["inferior_rim_ratio"],
            "isnt":        features["isnt_violation"],
            "disc_area_px": features["disc_area_px"],
            "cup_area_px":  features["cup_area_px"],
            "yolo_conf":   result.get("yolo_conf"),
            "breakdown":   breakdown,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Run directly for local testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
