"""
Inference directly on the .pt checkpoint (native ultralytics runtime).
Slowest of the three, but zero conversion needed — good for sanity-checking
that the other two stages produce matching results.

Install:
    pip install ultralytics

Usage:
    python 2_pt_inference.py --weights yolo11n.pt --source test.jpg
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--source", type=str, default="test.jpg", help="image, video, or folder path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="0", help="'0' for first GPU, 'cpu' for CPU")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source, conf=args.conf, iou=args.iou, device=args.device
    )

    for r in results:
        r.save(filename="output_pt.jpg")
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"{model.names[cls_id]:<15} conf={conf:.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
