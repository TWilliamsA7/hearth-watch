"""Relay the Jetson's camera stream (RTP/H.264 from run_edge.py --stream) into the dashboard.

Decodes the stream with OpenCV's FFmpeg and posts the newest frame as a JPEG to the server's
local camera preview (/camera-preview/<room>/<source>), the same place the laptop webcam client
uses, so the dashboard's Live camera view shows it. Nothing is recorded; only the latest frame
is kept. The dashboard draws the skeletons and posture labels from the observations.

    python -m hearth.perception.stream_relay --backend http://127.0.0.1:8000 --source jetson-01
"""
import argparse
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from .live_io import LatestPreview

# Low latency: no input buffering or probing, drop late frames.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "protocol_whitelist;file,udp,rtp|fflags;nobuffer|flags;low_delay|"
                      "probesize;32|analyzeduration;0|max_delay;0")

SDP = """v=0
o=- 0 0 IN IP4 {jetson}
s=Jetson camera
c=IN IP4 {laptop}
t=0 0
m=video {port} RTP/AVP 96
a=rtpmap:96 H264/90000
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--room", default="room-01")
    ap.add_argument("--source", default="jetson-01")
    ap.add_argument("--port", type=int, default=1234, help="UDP port the Jetson streams to")
    ap.add_argument("--laptop-ip", default="192.168.55.100")
    ap.add_argument("--jetson-ip", default="192.168.55.1")
    ap.add_argument("--max-fps", type=float, default=8.0, help="preview uploads per second")
    args = ap.parse_args(argv)

    import cv2

    sdp = Path(tempfile.gettempdir()) / f"hearth-{args.source}.sdp"
    sdp.write_text(SDP.format(jetson=args.jetson_ip, laptop=args.laptop_ip, port=args.port), encoding="ascii")
    url = args.backend.rstrip("/") + f"/camera-preview/{quote(args.room, safe='')}/{quote(args.source, safe='')}"
    opener = build_opener(ProxyHandler({}))

    def send(frame):
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            request = Request(url, data=jpeg.tobytes(), headers={"Content-Type": "image/jpeg"}, method="POST")
            with opener.open(request, timeout=1) as reply:
                reply.read()

    preview = LatestPreview(send)
    print(f"[relay] waiting for {args.source}'s stream on udp {args.port}", flush=True)
    try:
        while True:  # the Jetson may start later or restart: reopen until stopped
            capture = cv2.VideoCapture(str(sdp), cv2.CAP_FFMPEG)
            last, frames = 0.0, 0
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                if frames == 1:
                    print(f"[relay] {args.source} stream: {frame.shape[1]}x{frame.shape[0]}", flush=True)
                now = time.monotonic()
                if now - last >= 1 / args.max_fps:
                    last = now
                    preview.submit(frame)
            capture.release()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        preview.close()
        try:
            with opener.open(Request(url, method="DELETE"), timeout=1) as reply:
                reply.read()
        except OSError:
            pass


if __name__ == "__main__":
    main()
