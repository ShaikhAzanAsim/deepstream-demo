#!/usr/bin/env python3
"""
Programmatic DeepStream inference — use this when you need to pull detection
metadata into your own app (e.g. feed boxes into another service) instead of
just watching a rendered stream.

For most "just run it fast" use cases, prefer:
    deepstream-app -c 6_deepstream_app_config.txt
which does everything below via config only, no code needed.

Requirements (Linux only — DeepStream does not run on Windows):
    - NVIDIA DeepStream SDK installed and `deepstream-app` already working
    - deepstream_python_apps bindings (pyds) installed
    - 5_config_infer_primary_yolo11.txt pointing at your built .engine,
      in the same directory as this script (or edit CONFIG_FILE below)

Usage:
    python3 7_deepstream_python_infer.py file:///path/to/video.mp4
    python3 7_deepstream_python_infer.py rtsp://<camera-ip>:554/stream
"""
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

CONFIG_FILE = "5_config_infer_primary_yolo11.txt"


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            rect = obj_meta.rect_params
            print(
                f"frame={frame_meta.frame_num} class={obj_meta.obj_label} "
                f"conf={obj_meta.confidence:.2f} "
                f"box=({rect.left:.0f},{rect.top:.0f},{rect.width:.0f},{rect.height:.0f})"
            )
            # ^ this is the hook point — send obj_meta wherever you need it
            # instead of / in addition to printing it.
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def build_pipeline(source_uri):
    Gst.init(None)
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make("uridecodebin", "source")
    source.set_property("uri", source_uri)

    streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
    streammux.set_property("width", 1920)
    streammux.set_property("height", 1080)
    streammux.set_property("batch-size", 1)
    streammux.set_property("batched-push-timeout", 40000)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", CONFIG_FILE)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")  # swap for fakesink if headless

    for e in (source, streammux, pgie, nvvidconv, nvosd, sink):
        pipeline.add(e)

    def on_pad_added(_, new_pad):
        caps = new_pad.get_current_caps() or new_pad.query_caps()
        if not caps.to_string().startswith("video"):
            return  # skip audio pads from uridecodebin
        sinkpad = streammux.request_pad_simple("sink_0")
        new_pad.link(sinkpad)

    source.connect("pad-added", on_pad_added)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    osd_sink_pad = nvosd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    return pipeline


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <uri>   e.g. file:///path/video.mp4 or rtsp://...")
        sys.exit(1)

    pipeline = build_pipeline(sys.argv[1])
    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def bus_call(_, message):
        if message.type == Gst.MessageType.EOS:
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}: {debug}")
            loop.quit()
        return True

    bus.connect("message", bus_call)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
