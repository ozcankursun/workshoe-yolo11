# Work Shoe Detection — Short Report

## 1) Dataset
- Source(s): (Roboflow link or your own video frames)
- Size (images): train / val / test = ?, ?, ?
- Class(es): work_shoe (1 class)
- Notes: imbalance? label noise?

## 2) Training
- Models: yolo11n / yolo11s / yolo11m (state what you tried)
- Hyperparams: epochs, imgsz, batch, seed
- Hardware: CPU/GPU

## 3) Results
| Model   | Params (M) | mAP50 | mAP50-95 | Precision | Recall | FPS (640) |
|---------|------------|-------|----------|-----------|--------|-----------|
| yolo11n |            |       |          |           |        |           |
| yolo11s |            |       |          |           |        |           |
| yolo11m |            |       |          |           |        |           |

- Trade-offs discussion (size vs accuracy).
- Add inference images from `runs/predict/workshoes/`

## 4) Challenges
- e.g., small objects, occlusion, domain shift, annotation quality

## 5) Reproducibility
- `requirements.txt`, `configs/*.yaml`, `seed=42`
- Command to reproduce your best model:
```bash
python scripts/train.py --model yolo11n --data configs/data.yaml --epochs 50
```