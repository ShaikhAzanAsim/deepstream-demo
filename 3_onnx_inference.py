"""
Inference on the exported .onnx file using ONNX Runtime — no ultralytics,
no DeepStream. Useful for validating the export before you touch TensorRT.

YOLO11's ONNX output is a single tensor of shape (1, 4 + num_classes, 8400):
  - 4 box values per anchor: cx, cy, w, h (already decoded, in the padded/
    resized 640x640 input space — no separate objectness score, class
    scores are already sigmoid-activated)
  - num_classes scores per anchor

So we still have to do confidence filtering + NMS + un-letterbox ourselves.

Install:
    pip install onnxruntime-gpu opencv-python numpy
    (use onnxruntime instead of onnxruntime-gpu if you don't have CUDA here)

Usage:
    python 3_onnx_inference.py --onnx yolo11n.onnx --source test.jpg
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]  # replace with your own labels.txt contents if you fine-tuned the model


def letterbox(img, new_size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def preprocess(img, size=640):
    padded, r, pad_x, pad_y = letterbox(img, size)
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW, 0-1
    blob = np.expand_dims(np.ascontiguousarray(blob), axis=0)
    return blob, r, pad_x, pad_y


def postprocess(pred, r, pad_x, pad_y, orig_w, orig_h, conf_thres=0.25, iou_thres=0.45):
    pred = pred[0].T  # (8400, 4+num_classes)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    keep = confidences > conf_thres
    boxes_xywh, class_ids, confidences = boxes_xywh[keep], class_ids[keep], confidences[keep]
    if len(boxes_xywh) == 0:
        return []

    # center-xywh -> top-left xywh (still in the letterboxed 640x640 space)
    x = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    w = boxes_xywh[:, 2]
    h = boxes_xywh[:, 3]
    nms_boxes = np.stack([x, y, w, h], axis=1)

    idxs = cv2.dnn.NMSBoxes(nms_boxes.tolist(), confidences.tolist(), conf_thres, iou_thres)

    results = []
    for i in np.array(idxs).flatten():
        bx, by, bw, bh = nms_boxes[i]
        # undo letterbox padding, then rescale to the original image size
        x1 = max(0, (bx - pad_x) / r)
        y1 = max(0, (by - pad_y) / r)
        x2 = min(orig_w, (bx + bw - pad_x) / r)
        y2 = min(orig_h, (by + bh - pad_y) / r)
        cid = int(class_ids[i])
        results.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(confidences[i]),
                "class_id": cid,
                "class_name": COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else str(cid),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="yolo11n.onnx")
    parser.add_argument("--source", type=str, default="test.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    session = ort.InferenceSession(
        args.onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    img = cv2.imread(args.source)
    orig_h, orig_w = img.shape[:2]
    blob, r, pad_x, pad_y = preprocess(img)

    outputs = session.run(None, {input_name: blob})
    detections = postprocess(outputs[0], r, pad_x, pad_y, orig_w, orig_h, args.conf, args.iou)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.putText(img, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f'{det["class_name"]:<15} conf={det["confidence"]:.2f}  box=({x1},{y1},{x2},{y2})')

    cv2.imwrite("output_onnx.jpg", img)
    print(f"Found {len(detections)} objects -> output_onnx.jpg")


if __name__ == "__main__":
    main()
