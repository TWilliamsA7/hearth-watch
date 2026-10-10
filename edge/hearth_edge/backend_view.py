"""Polls the team backend for its own view of this source, for the video preview panel only.

The backend (Person 2's monitor) decides posture and alerts; the edge never does. This asks it,
about twice a second on a background thread, what it currently thinks:
    GET /sources                 -> online?, posture (standing / fallen / unknown), stale?
    GET /alerts?source=<source>  -> open / acknowledged alerts for this camera
Nothing here changes what the edge sends. If the backend has no such pages (the stand-in
stub_backend.py) or can't be reached, the panel just says the backend view is unavailable.
Standard library only (Python 3.6).
"""

import json
import threading
import time
from urllib import parse as urlparse
from urllib import request as urlrequest

from hearth_edge.contract import parse_utc_iso


def source_summary(sources, source):
    """This source's entry from GET /sources, reduced to what the panel shows, or None."""
    for s in sources or []:
        if s.get("source") == source:
            return {"online": bool(s.get("online")),
                    "posture": s.get("posture") or "unknown",
                    "stale": bool(s.get("observation_stale", False))}
    return None


def alert_summary(alerts, now):
    """Unresolved alerts, newest first, as [(kind, status, age_s or None, accepted_by or None)]."""
    out = []
    for a in alerts or []:
        if a.get("status") == "resolved":
            continue
        try:
            age = now - parse_utc_iso(a["opened_at"])
        except (KeyError, TypeError, ValueError):
            age = None
        out.append((a.get("kind", "alert"), a.get("status", "?"), age, a.get("accepted_by_name")))
    out.sort(key=lambda x: float("inf") if x[2] is None else x[2])
    return out


class BackendView(object):
    def __init__(self, base_url, source, interval_s=0.5, timeout_s=1.0, clock=time.time):
        self.base = base_url.rstrip("/")
        self.source = source
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state = {"reachable": None}
        self._thread = None

    def _get(self, path):
        with urlrequest.urlopen(self.base + path, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def poll_once(self):
        try:
            sources = self._get("/sources")
            alerts = self._get("/alerts?source=" + urlparse.quote(self.source))
            state = {"reachable": True, "source": source_summary(sources, self.source),
                     "alerts": alert_summary(alerts, self.clock()), "t": self.clock()}
        except Exception as e:  # unreachable, or a backend without these pages (the stub)
            code = getattr(e, "code", None)
            state = {"reachable": False, "error": "HTTP %d" % code if code else type(e).__name__}
        with self._lock:
            self._state = state
        return state

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def start(self):
        def run():
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(self.interval_s)
        self._thread = threading.Thread(target=run, name="backend-view")
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
