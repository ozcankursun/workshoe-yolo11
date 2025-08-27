import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolo11n", choices=["yolo11n","yolo11s","yolo11m","yolo11l","yolo11x"])
    ap.add_argument("--cfg", type=str, default=None, help="Optional: path to a train YAML (overrides defaults).")
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model_path = f"{args.model}.pt" if not args.model.endswith(".pt") else args.model
    model = YOLO(model_path)

    # prefer smaller models when possible
    print(f"Training model: {model_path}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        seed=args.seed,
        verbose=True,
        plots=True,
        # You can add more hyperparams or pass a YAML with --cfg
    )
    print(results)

if __name__ == "__main__":
    main()