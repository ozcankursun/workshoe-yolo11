# Work Shoe Detection — YOLOv11

This repo trains YOLOv11 to detect work shoes. Includes configs, scripts, and a report template.

## Quickstart
```bash
# 1) Create & activate env
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Put your dataset in data/workshoes with the YOLO layout:
# data/workshoes/
#   images/train, labels/train
#   images/val,   labels/val
#   images/test,  labels/test

# 4) Edit configs/data.yaml (paths) as needed.

# 5) Train (nano example)
python scripts/train.py --model yolo11n --data configs/data.yaml --epochs 50

# 6) Evaluate
python scripts/eval.py --weights runs/detect/train/weights/best.pt --data configs/data.yaml

# 7) Export ONNX
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
```