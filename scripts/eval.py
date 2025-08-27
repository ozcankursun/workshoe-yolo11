import argparse, json, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained weights (e.g., runs/detect/train/weights/best.pt)")
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split="val", save_json=True)  # returns a Results object
    # Extract common metrics
    out = {
        "precision": float(metrics.results_dict.get("metrics/precision(B)", 0.0)),
        "recall":    float(metrics.results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50":     float(metrics.results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95":  float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "nc":        metrics.results_dict.get("metrics/NC", None),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/val_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()