"""Optional local recognition transport, sharing the fall camera's exact frames."""
import base64
import json
from urllib.parse import urlparse
from urllib.request import Request, build_opener, ProxyHandler


class PatientClient:
    def __init__(self, backend, room):
        if urlparse(backend).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Patient demo frames must stay on the same machine as the backend")
        self.backend, self.room, self.session = backend.rstrip("/"), room, None
        self.opener = build_opener(ProxyHandler({}))  # Camera frames never use an environment proxy.

    def request(self, path, body=None, method="POST"):
        request = Request(self.backend + "/face-demo" + path,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type":"application/json", "X-Hearth-Face-Demo":"1"}, method=method)
        with self.opener.open(request, timeout=3) as response:
            return json.load(response)

    def observe(self, frame, observation, *, body=None):
        import cv2
        if self.session is None:
            self.session = self.request("/sessions", {"room_id":self.room})["session_id"]
        height, width = frame.shape[:2]
        scale = min(1., 960 / width, 720 / height)
        resized = cv2.resize(frame, (round(width*scale), round(height*scale)))
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ValueError("Cannot encode camera frame")
        payload = {"session_id":self.session, "frame":base64.b64encode(encoded).decode(),
                   "observation":observation.model_dump(mode="json") if hasattr(observation, "model_dump") else observation}
        if body is not None:
            payload["body"] = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        return self.request("/observe", payload)

    def close(self):
        if self.session:
            session, self.session = self.session, None
            try:
                self.request("/sessions/" + session, method="DELETE")
            except (OSError, ValueError):
                pass
