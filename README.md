# YOLO11n → ONNX → TensorRT engine → DeepStream

## Platform note (you're on Windows)

DeepStream itself only runs on **Linux** — Ubuntu x86 with an NVIDIA dGPU,
or a Jetson (JetPack). It is not supported natively on Windows, and WSL2 is
not an officially supported path for DeepStream (CUDA-only workloads work in
WSL2; DeepStream's GStreamer/multimedia plugins generally don't).

Practical split:
- Steps 1 (export to ONNX) and the ONNX Runtime test in step 1b can run on
  your Windows box — it's plain PyTorch/ONNX Runtime.
- Steps 2–3 (build the `.engine`, run DeepStream) need to happen on a Linux
  machine or Jetson device — either a Linux box you deploy to, or the
  official DeepStream Docker container.

## Why DeepStream needs its own ONNX export

`nvinfer` (DeepStream's inference plugin) only knows how to decode a handful
of built-in output layouts. YOLO11's raw output tensor — box + class scores
per anchor, no NMS — isn't one of them, so DeepStream needs a **custom
bbox-parsing library** to turn that tensor into boxes. The community
standard for this is [marcoslucianops/DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo),
which ships both the custom parser (`nvdsinfer_custom_impl_Yolo`, compiled
as a `.so`) and a matching export script that lays out the ONNX graph the
way that parser expects.

So there are effectively two ONNX exports in this pipeline:
- **Generic ONNX** (`1_export_to_onnx.py`, plain `ultralytics` export) — for
  `3_onnx_inference.py` and for learning/testing TensorRT conversion.
- **DeepStream ONNX** (DeepStream-Yolo's `export_yolo11.py`) — the one you
  actually build the `.engine` from for DeepStream. Steps below.

## Step 1 — Export .pt → ONNX

```bash
pip install ultralytics
python 1_export_to_onnx.py --weights yolo11n.pt --imgsz 640 --opset 12
```

Test it standalone before moving on:
```bash
pip install onnxruntime-gpu opencv-python numpy   # or onnxruntime (CPU) if no local CUDA
python 3_onnx_inference.py --onnx yolo11n.onnx --source test.jpg
```
(`2_pt_inference.py` runs the same test directly on the `.pt` file, if you
want a baseline to compare against.)

## Step 2 — Get the DeepStream-flavored ONNX + build the engine

On the Linux/Jetson target:

```bash
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cp DeepStream-Yolo/utils/export_yolo11.py ultralytics/
cd ultralytics
pip3 install -e .
pip3 install onnx onnxslim onnxruntime

# use your fine-tuned weights, or download a stock checkpoint:
# wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

python3 export_yolo11.py -w yolo11n.pt --dynamic --simplify
cp yolo11n.onnx ../DeepStream-Yolo/
```

Build the TensorRT engine (do this on the exact machine/GPU you'll deploy
on — engines aren't portable across GPU architectures or TensorRT versions):
```bash
cd ../DeepStream-Yolo
../4_build_engine.sh yolo11n.onnx yolo11n_fp16.engine
```
DeepStream can also build the engine itself on first run if you skip this
and just point it at the `.onnx` — but that first run can take 10+ minutes,
so pre-building is worth it for anything production-facing.

Compile DeepStream-Yolo's custom parser lib (match `CUDA_VER` to your
DeepStream version — table below):
```bash
export CUDA_VER=12.6   # example: DeepStream 7.1 on x86
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

| DeepStream | Ubuntu (x86) | CUDA (x86) | TensorRT (x86) |
|---|---|---|---|
| 8.0 | 24.04 | 12.8 | 10.9 |
| 7.1 | 22.04 | 12.6 | 10.4 |
| 7.0 / 6.4 | 22.04 | 12.2 | 8.6 |
| 6.3 | 20.04 | 12.1 | 8.5 |
| 6.2 | 20.04 | 11.8 | 8.5 |

(Jetson uses JetPack's bundled CUDA/TensorRT instead — see the DeepStream-Yolo
README's requirements section for the matching table.)

## Step 3 — Wire up DeepStream

Copy `5_config_infer_primary_yolo11.txt` and `6_deepstream_app_config.txt`
into the `DeepStream-Yolo` folder (they already reference `libnvdsinfer_custom_impl_Yolo.so`,
`yolo11n.onnx`, and `yolo11n_fp16.engine` — adjust paths/class count if your
model differs), then edit `6_deepstream_app_config.txt`'s `[source0]` to
point at your video file or RTSP camera, and run:

```bash
deepstream-app -c 6_deepstream_app_config.txt
```

That's the "just run it" path. If you need to pull detections into your own
code (e.g. push into another service) instead of watching a rendered
stream, use `7_deepstream_python_infer.py` — it builds the same pipeline
programmatically and prints each detection from a probe callback; swap the
`print()` for whatever you need. Requires the `deepstream_python_apps`
(`pyds`) bindings installed alongside DeepStream.

## Files in this bundle

| File | Runs on | Purpose |
|---|---|---|
| `1_export_to_onnx.py` | anywhere | .pt → generic .onnx |
| `2_pt_inference.py` | anywhere (GPU optional) | inference on .pt |
| `3_onnx_inference.py` | anywhere | inference on .onnx via ONNX Runtime |
| `4_build_engine.sh` | Linux/Jetson, TensorRT installed | .onnx → .engine via trtexec |
| `5_config_infer_primary_yolo11.txt` | DeepStream | nvinfer config for the engine |
| `6_deepstream_app_config.txt` | DeepStream | top-level pipeline config |
| `7_deepstream_python_infer.py` | DeepStream + pyds | programmatic .engine inference |
