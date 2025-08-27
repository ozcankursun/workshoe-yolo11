# Work Shoe Detection with YOLOv11

This repository trains and evaluates **YOLOv11** to detect **work shoes**. It includes modular scripts, configs, a short report, and an optional Jupyter Notebook for full reproducibility.

> **TL;DR** — Final pick: **YOLOv11n (50 epochs)** for the best speed/accuracy on CPU.  
> **Val mAP@50–95:** 0.669 · **Test mAP@50–95:** 0.553 · **~FPS (CPU, 512):** 8–12

---

## Table of Contents
- [Work Shoe Detection with YOLOv11](#work-shoe-detection-with-yolov11)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
    - [1) Create \& activate env](#1-create--activate-env)
    - [2) Install dependencies](#2-install-dependencies)
  - [Dataset Setup](#dataset-setup)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Results](#results)
  - [ONNX Export \& Edge Deployment](#onnx-export--edge-deployment)
  - [Reproducibility](#reproducibility)
  - [Troubleshooting](#troubleshooting)
  - [License \& Ethics](#license--ethics)

---

## Features
- Small, clean **CLI scripts**: `train.py`, `eval.py`, `infer.py`, `export_onnx.py`
- **Config-first**: `configs/data.yaml` controls dataset paths
- **Trade-off study**: YOLOv11 **nano** vs **small**
- **Outputs** saved under `runs/` and key metrics under `outputs/`
- **Report**: `reports/REPORT.md` (with embedded figures)
- (Optional) **Notebook** for end-to-end runs

---

## Repository Structure
```
configs/
  data.yaml
reports/
  REPORT.md
  figures/              # put selected prediction images here
scripts/
  train.py
  eval.py
  infer.py
  export_onnx.py
data/workshoes/
  images/{train,val,test}
  labels/{train,val,test}
outputs/
  val_metrics.json
  val_comparison.csv
runs/
  detect/..., predict/...   # auto-created by Ultralytics
```

---

## Getting Started

### 1) Create & activate env
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note (Windows ONNX export):** If you later export ONNX, Python **3.11** with **NumPy 1.26.x** works best alongside `onnxruntime`. See [Troubleshooting](#troubleshooting).

---

## Dataset Setup

Place your dataset in YOLO format:

```
data/workshoes/
  images/train   labels/train
  images/val     labels/val
  images/test    labels/test
```

Edit `configs/data.yaml` if needed:
```yaml
path: ../data/workshoes
train: images/train
val: images/val
test: images/test
names:
  0: work_shoe
```

**Tip:** If your split is `valid`, rename the directory to `val` (Ultralytics expects `val`).

---

## Training

**Nano (recommended baseline)**
```bash
python scripts/train.py --model yolo11n --data configs/data.yaml --epochs 50 --batch 8 --imgsz 512
```

**Small (comparison)**
```bash
python scripts/train.py --model yolo11s --data configs/data.yaml --epochs 50 --batch 8 --imgsz 512
```

---

## Evaluation
```bash
python scripts/eval.py --weights runs/detect/<train_run>/weights/best.pt --data configs/data.yaml
```

> To evaluate **test** split programmatically, see the Notebook or use:
```bash
python - << "PY"
from ultralytics import YOLO
m = YOLO("runs/detect/<train_run>/weights/best.pt")
print(m.val(data="configs/data.yaml", split="test", save_json=True).results_dict)
PY
```

---

## Inference
```bash
python scripts/infer.py --weights runs/detect/<train_run>/weights/best.pt --source data/workshoes/images/val --conf 0.35
# outputs under runs/predict/<name>/
```

---

## Results

**Validation (val)**

| Model     | Epochs | Precision | Recall  | mAP@50  | mAP@50–95 | Inference (ms/img) | ~FPS |
|-----------|-------:|----------:|--------:|--------:|----------:|-------------------:|-----:|
| YOLOv11n  |   10   | 0.8754    | 0.7917  | 0.9038  | 0.6024    | 119.29             | 8.4  |
| **YOLOv11n** | **50** | **0.9581**  | **0.9533**| **0.9876**| **0.6695** | **121.29**           | **8.2** |
| YOLOv11s  |   50   | 1.0000    | 0.8709  | 0.9698  | 0.6303    | 229.01             | 4.4  |

**Test (final: YOLOv11n @ 50e)**  
Precision **0.9974** · Recall **0.8889** · mAP@50 **0.9197** · mAP@50–95 **0.5530**

> See **`reports/REPORT.md`** for embedded figures (val/test predictions and curves).

---

## ONNX Export & Edge Deployment

Export final model to ONNX (portable for CPU/Edge):
```bash
python scripts/export_onnx.py --weights runs/detect/<train_run>/weights/best.pt --half
# output: runs/detect/<train_run>/weights/best.onnx (~11 MB)
```

Quick validation & ONNX inference:
```bash
python - << "PY"
from ultralytics import YOLO
YOLO("runs/detect/<train_run>/weights/best.onnx").predict(
    source="data/workshoes/images/val", conf=0.35, save=True)
PY
```

> Consider hosting large artifacts (ONNX/PT) via **GitHub Releases** or **Git LFS**.

---

## Reproducibility
- **Seed:** 42  
- **Environment:** Python 3.10–3.13 (training); Python **3.11** recommended for ONNX export  
- **Dependencies:** `requirements.txt`  
- **Artifacts:** `outputs/` (metrics CSV/JSON), `runs/` (training/prediction outputs)  
- Optional: use Weights & Biases or MLflow for experiment tracking

---

## Troubleshooting

**1) `FileNotFoundError: ... images/val not found`**  
Ensure split directory names are exactly `train/val/test` and paths in `configs/data.yaml` match.

**2) ONNX export errors on Python 3.13 (NumPy 2.x)**  
Use a **clean Python 3.11 venv** and install:
```bash
pip install numpy==1.26.4 onnx==1.16.2 onnxruntime==1.18.0 ultralytics==8.3.186
```
Then re-run export.

**3) Slow inference on CPU**  
Keep `imgsz=512`, prefer **YOLOv11n**, and tune `--conf` (e.g., 0.35) for recall vs. FP.

---

## License & Ethics
- Source video used for **personal/educational** purposes.  
- If sharing dataset or weights publicly/commercially, verify original content’s **license** and obtain permissions.