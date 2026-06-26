# 👁️Crowded Disc Detector

An automated screening tool for **Non-Arteritic Anterior Ischemic Optic Neuropathy (NAION)** risk assessment from fundus photographs using deep learning and clinical feature extraction.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Dagshub-blue.svg)](https://dagshub.com/jani2904-sys/NAION-Risk-Analyzer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🔬 What is NAION?

Non-Arteritic Anterior Ischemic Optic Neuropathy (NAION) is the most common acute optic neuropathy in adults over 50, caused by ischemia of the optic nerve head. A key anatomical risk factor is a **crowded disc** — a small optic cup with little space for the nerve fibres, making the disc susceptible to ischemic damage.

This tool analyzes the structural anatomy of the optic disc and cup from fundus photographs to compute a NAION risk score based on published clinical criteria.

---

## 🏗️ Pipeline Architecture

```
Fundus Image
      ↓
┌─────────────────────────┐
│  YOLOv8n                │  Disc localization
│  mAP@50 = 0.995         │  → bounding box crop
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  UNet + EfficientNet-b4 │  Disc + cup segmentation
│  Disc Dice  = 0.941     │  → binary masks
│  Cup Dice   = 0.848     │
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Clinical Feature       │  vCDR, CDI, rim thickness,
│  Extraction             │  inferior rim, ISNT rule,
│                         │  disc-fovea distance
└─────────────────────────┘
      ↓
┌─────────────────────────┐
│  Rule-Based Risk Score  │  Literature-based thresholds
│  0 – 100                │  → Low / Moderate / High
└─────────────────────────┘
```

---

## 📊 Model Performance

| Model | Architecture | Task | Metric | Score |
|---|---|---|---|---|
| Disc Localizer | YOLOv8n | Detection | mAP@50 | 0.995 |
| Disc Localizer | YOLOv8n | Detection | Precision | 0.997 |
| Disc Localizer | YOLOv8n | Detection | Recall | 0.993 |
| Segmentation | UNet + EfficientNet-b4 + scSE | Disc Seg | Dice | 0.941 |
| Segmentation | UNet + EfficientNet-b4 + scSE | Cup Seg | Dice | 0.848 |

Trained on [REFUGE2](https://refuge.grand-challenge.org/) — 800 training images, 400 validation, 400 test images with optic disc/cup segmentation masks.

---

## 🩺 NAION Risk Criteria

Risk score (0–100) computed from 6 rules based on published NAION literature:

| Rule | Feature | Threshold | Weight | Clinical Basis |
|---|---|---|---|---|
| 1 | vCDR | < 0.5 | 35 pts | Primary crowded disc criterion (Hayreh 2009) |
| 2 | CDI | > 0.78 | 20 pts | Structural crowding index (Contreras 2018) |
| 3 | Rim thickness ratio | < 0.12 | 20 pts | Thin neuroretinal rim |
| 4 | Area CDR | < 0.20 | 15 pts | Small cup relative to disc |
| 5 | Disc roundness | > 0.93 | 5 pts | NAION disc morphology |
| 6 | Inferior rim ratio | < 0.90 | 20 pts | Preferential inferior damage (Arnold 2003) |
| Bonus | ISNT rule violation | — | +5 pts | Inferior not thickest sector |

**Score interpretation:**
- 🟢 **0–39**: Low risk
- 🟠 **40–64**: Moderate risk
- 🔴 **65–100**: High risk

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- CUDA GPU recommended (CPU inference supported but slow)
- Dagshub account for MLflow model registry

### Installation

```bash
# Clone the repository
git clone https://github.com/jani2904-sys/fundus-naion-risk.git
cd fundus-naion-risk

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your Dagshub token and paths
```

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/jani2904-sys/NAION-Risk-Analyzer.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
YOLO_MODEL_PATH=weights/yolov8n_disc_localizer.pt
REFUGE2_BASE=/path/to/REFUGE2
OUTPUT_DIR=naion_results
```

### Model Weights

Download pre-trained weights:

| Model | File | Size | Source |
|---|---|---|---|
| YOLO Disc Localizer | `weights/yolov8n_disc_localizer.pt` | ~6MB | Included in repo |
| UNet Segmentation | Auto-loaded from MLflow | ~80MB | Dagshub registry |

The segmentation model loads automatically from the Dagshub MLflow registry — no manual download needed.

---

## 💻 Usage

### Run Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser, upload a fundus image, and get instant results.

### Run FastAPI Service

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

**Example API call:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@fundus_image.jpg"
```

**Example response:**
```json
{
  "risk_score": 42,
  "risk_level": "Moderate",
  "vCDR": 0.387,
  "CDI": 0.812,
  "rim_ratio": 0.094,
  "inf_rim": 0.881,
  "isnt": true,
  "breakdown": {
    "vCDR < 0.4 (moderately crowded)": 20,
    "CDI > 0.78 (crowded disc)": 12,
    "Rim ratio < 0.12 (borderline rim)": 7,
    "ISNT rule violated": 5
  }
}
```

### Run with Docker

```bash
docker build -t naion-analyzer .
docker run -p 8000:8000 -p 8501:8501 --env-file .env naion-analyzer
```

### Run Feature Extraction on Test Set

```bash
python inference/feature_extraction.py
```

Processes all test images and saves results to `naion_results/`:
- Per-image result cards (overlay + risk score + feature table)
- `naion_risk_scores.csv` with all features

---

## 📁 Repository Structure

```
fundus-naion-risk/
│
├── models/
│   ├── train.py                  ← EfficientNet-b4 UNet training
│   ├── train_yolo.py             ← YOLOv8n disc localizer training
│   └── dataset.py                ← REFUGE2 dataset class
│
├── inference/
│   ├── inference.py              ← Segmentation inference pipeline
│   └── feature_extraction.py    ← Clinical features + risk score
│
├── evaluation/
│   ├── overlay.py                ← Contour visualization vs GT
│   └── pr_curve.py               ← Threshold analysis
│
├── app/
│   ├── app.py                    ← FastAPI service
│   ├── streamlit_app.py          ← Streamlit demo
│   └── start.sh                  ← Docker startup script
│
├── weights/
│   └── yolov8n_disc_localizer.pt ← YOLO weights
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔬 Training

### Train Segmentation Model

```bash
python models/train.py
```

Trains EfficientNet-b4 UNet with:
- Weighted BCE + Dice loss (cup weight = 3.0)
- scSE attention decoder
- Input size 384×384
- Early stopping on validation Cup Dice

### Train YOLO Disc Localizer

```bash
python models/train_yolo.py
```

Automatically generates YOLO labels from REFUGE2 masks and trains YOLOv8n for 50 epochs.

All training runs are tracked in MLflow on Dagshub:
🔗 [View Experiments](https://dagshub.com/jani2904-sys/NAION-Risk-Analyzer.mlflow)

---

## 📚 References

1. Hayreh SS. *Ischemic optic neuropathy.* Progress in Retinal and Eye Research. 2009.
2. Contreras I, et al. *Crowded disc and NAION risk.* 2018.
3. Arnold AC. *Pathogenesis of nonarteritic anterior ischemic optic neuropathy.* Journal of Neuro-Ophthalmology. 2003.
4. Orlando JI, et al. *REFUGE2 Challenge.* arXiv:2202.08994. 2022.

---

## ⚠️ Disclaimer

This tool is intended for **research purposes only**. It has not been validated for clinical use and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified ophthalmologist for diagnosis and management of optic neuropathy.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Citation

If you use this work in your research, please cite:

```bibtex
@software{naion_risk_analyzer_2026,
  author = {Jani},
  title  = {NAION Risk Analyzer: Automated Fundus-Based Screening},
  year   = {2026},
  url    = {https://github.com/jani2904-sys/fundus-naion-risk}
}
```

*Paper in preparation.*
