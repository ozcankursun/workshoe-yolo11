import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    ap.add_argument("--half", action="store_true", help="Export FP16")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(format="onnx", half=args.half, opset=12, dynamic=True)
    print("Exported ONNX to 'weights/' or model's default export dir.")

if __name__ == "__main__":
    main()