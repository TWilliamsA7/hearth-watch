"""Relay the Jetson's camera stream (RTP/H.264 from run_edge.py --stream) into the dashboard.

Decodes the stream with PyAV (FFmpeg; OpenCV's build can't open RTP from an SDP file) and posts the newest frame as a JPEG to the server's
local camera preview (/camera-preview/<room>/<source>), the same place the laptop webcam client
uses, so the dashboard's Live camera view shows it. Nothing is recorded; only the latest frame
is kept. The dashboard draws the skeletons and posture labels from the observations.

    python -m hearth.perception.stream_relay --backend http://127.0.0.1:8000 --source jetson-01
"""
import argparse
from pathlib import Path
import tempfile
import time
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener

from .live_io import LatestPreview

# The SDP file names an RTP stream, so those protocols must be allowed; low latency: no input
# buffering, little probing.
OPTIONS = {"protocol_whitelist": "file,udp,rtp", "fflags": "nobuffer", "flags": "low_delay",
           "probesize": "32768", "analyzeduration": "0", "max_delay": "0", "reorder_queue_size": "0"}

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

    import av
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
            last, frames = 0.0, 0
            try:
                with av.open(str(sdp), options=OPTIONS, timeout=(10, 5)) as container:
                    for decoded in container.decode(video=0):
                        now = time.monotonic()
                        if now - last < 1 / args.max_fps:
                            continue  # decode every frame (H.264 needs it), upload a few
                        last = now
                        frame = decoded.to_ndarray(format="bgr24")
                        frames += 1
                        if frames == 1:
                            print(f"[relay] {args.source} stream: {frame.shape[1]}x{frame.shape[0]}", flush=True)
                        preview.submit(frame)
            except (av.FFmpegError, OSError) as error:
                if frames:
                    print(f"[relay] {args.source} stream stopped ({error}); waiting for it again", flush=True)
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
