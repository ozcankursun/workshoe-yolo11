import argparse, os
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    ap.add_argument("--source", type=str, default="data/workshoes/images/val")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    res = model.predict(source=args.source, conf=args.conf, save=True, project="runs/predict", name="workshoes")
    print("Saved predictions to:", os.path.join("runs", "predict", "workshoes"))

if __name__ == "__main__":
    main()