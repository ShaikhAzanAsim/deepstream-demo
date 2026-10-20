"""
Step 1 — Export a YOLO11 .pt checkpoint to a generic ONNX file.

This ONNX file is what you use for:
  - standalone ONNX Runtime inference (2_onnx_inference.py)
  - building a TensorRT .engine with trtexec (4_build_engine.sh)

NOTE: For the DeepStream leg specifically, DeepStream-Yolo provides its own
export_yolo11.py (see README.md, "Why DeepStream needs its own ONNX export")
which produces an ONNX graph laid out for its custom bbox parser. Use that
one when you build the .engine that DeepStream will load.

Install:
    pip install ultralytics

Usage:
    python 1_export_to_onnx.py --weights yolo11n.pt --imgsz 640 --opset 12
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="path to .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size (square)")
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset. 12 is the safest choice for older TensorRT (<=8.5). "
        "Use 17 if you're on TensorRT 8.6+/10.x.",
    )
    parser.add_argument("--dynamic", action="store_true", help="export with dynamic batch size")
    parser.add_argument(
        "--no-simplify", action="store_true", help="skip onnx-simplifier (on by default)"
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=not args.no_simplify,
    )
    print(f"Saved ONNX model to: {onnx_path}")


if __name__ == "__main__":
    main()
