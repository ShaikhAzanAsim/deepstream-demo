#!/usr/bin/env bash
# Step 2 — Build a TensorRT .engine from the ONNX model.
#
# IMPORTANT: run this ON (or targeting the same GPU architecture + TensorRT
# version as) the machine that will actually run DeepStream. A .engine file
# is NOT portable across different GPUs or TensorRT versions — you'll have
# to rebuild it if you move it.
#
# trtexec ships with the TensorRT install / DeepStream container, so this
# needs to run on Linux (or inside the DeepStream docker image), not on
# Windows.
#
# Usage:
#   ./4_build_engine.sh yolo11n.onnx yolo11n_fp16.engine

set -e

ONNX_FILE=${1:-yolo11n.onnx}
ENGINE_FILE=${2:-yolo11n_fp16.engine}

# --- Static batch, FP16 (good default — ~2x throughput over FP32, minimal accuracy loss) ---
trtexec \
  --onnx="${ONNX_FILE}" \
  --saveEngine="${ENGINE_FILE}" \
  --fp16 \
  --memPoolSize=workspace:4096

echo "Saved engine to ${ENGINE_FILE}"

# --- Dynamic batch example (only if the ONNX was exported with --dynamic) ---
# trtexec \
#   --onnx="${ONNX_FILE}" \
#   --saveEngine="${ENGINE_FILE}" \
#   --fp16 \
#   --memPoolSize=workspace:4096 \
#   --minShapes=images:1x3x640x640 \
#   --optShapes=images:8x3x640x640 \
#   --maxShapes=images:16x3x640x640

# --- INT8 (fastest, needs a calibration cache first) ---
# See DeepStream-Yolo/docs/INT8Calibration.md for generating calib.cache
# trtexec \
#   --onnx="${ONNX_FILE}" \
#   --saveEngine="${ENGINE_FILE%.engine}_int8.engine" \
#   --int8 \
#   --calib=calib.cache \
#   --memPoolSize=workspace:4096

# NOTE ON FLAG NAMES: older TensorRT (<=8.3) uses "--workspace=4096" (MB)
# instead of "--memPoolSize=workspace:4096". Check `trtexec --help | grep -i workspace`
# if this errors out.
