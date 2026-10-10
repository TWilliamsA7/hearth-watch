"""Information panel on the optional video preview. DISPLAY ONLY: drawn on the stream, never sent.

    13.6 FPS | 10 obs/s | backend online                       04:58:12
    person: yes (12 keypoints) | face: yes 0.99 (0.3 s ago)
    torso 4 deg | head 0.21 | hip 0.55 | box 0.38 | drop +0.01/s
    backend: online | posture standing                 <- the backend's own view
    alerts: none open                                  <- (or: ALERT possible fall: open 4 s)
    ... head and hip dots, and a dashed "floor zone" line (aiming aid)

The edge makes no fall decision: lines 1-3 are its raw measurements, and the backend lines are
whatever the backend reports (see backend_view.py). The floor line marks the height the backend
requires head and hips to be below to count as floor level (backend/hearth/monitor/config.json
floor_head_y_min / floor_hip_y_min), to help aim the camera at the floor.
"""

WHITE = (255, 255, 255, 255)
GREY = (170, 170, 170, 255)
GREEN = (90, 230, 90, 255)
AMBER = (255, 190, 0, 255)
RED = (255, 60, 60, 255)
CYAN = (0, 220, 255, 255)


def _fmt(v, fmt):
    return "-" if v is None else fmt % v


def status_line(fps, obs_rate, mode, clock_str):
    """mode: 'DRY RUN', 'backend online', 'backend OFFLINE' or 'backend not tried'."""
    return "%s FPS | %s obs/s | %s    %s" % (_fmt(fps, "%.1f"), _fmt(obs_rate, "%.0f"), mode, clock_str)


def person_line(person_detected, n_keypoints, face, n_held=0, source=None):
    """face: None (no result yet / off) or (present, confidence, age_s); n_held: keypoints
    bridged by features.KeypointHold this frame; source: e.g. 'rotated CCW' when the pose
    came from a rotated poseNet pass."""
    extras = (", %d held" % n_held if n_held else "") + (", %s" % source if source else "")
    who = "yes (%d keypoints%s)" % (n_keypoints, extras) if person_detected else "no"
    if face is None:
        face_txt = "-"
    else:
        present, conf, age = face
        face_txt = "%s%s (%.1f s ago)" % ("yes" if present else "no",
                                          " %.2f" % conf if present and conf is not None else "", age)
    return "person: %s | face: %s" % (who, face_txt)


def features_line(feats):
    f = feats or {}
    return "torso %s | head %s | hip %s | box %s | drop %s" % (
        _fmt(f.get("torso_angle_deg"), "%.0f deg"), _fmt(f.get("head_y"), "%.2f"),
        _fmt(f.get("hip_y"), "%.2f"), _fmt(f.get("bbox_aspect"), "%.2f"),
        _fmt(f.get("vertical_velocity"), "%+.2f/s"))


def backend_lines(view, source):
    """The backend's own view as [(text, colour)]. view: BackendView.snapshot() or None."""
    if view is None:
        return []
    if view.get("reachable") is None:
        return [("backend view: waiting...", GREY)]
    if not view.get("reachable"):
        return [("backend view: unavailable (%s)" % view.get("error", "?"), GREY)]
    src = view.get("source")
    if src is None:
        return [("backend: no status for %s yet" % source, GREY)]
    posture = src["posture"]
    colour = {"standing": GREEN, "fallen": RED}.get(posture, GREY)
    text = "backend: %s | posture %s%s" % ("online" if src["online"] else "OFFLINE", posture,
                                          " (stale data)" if src["stale"] else "")
    lines = [(text, colour if src["online"] else AMBER)]
    alerts = view.get("alerts") or []
    if not alerts:
        lines.append(("alerts: none open", GREY))
    for kind, status, age, who, down in alerts[:2]:
        name = kind.replace("_", " ")
        by = " (%s)" % who if who else ""
        if down is not None:
            # Recovered: stop counting; the alert stays until a nurse resolves it on /nurses.
            back = "reconnected" if kind == "source_offline" else "up again"
            lines.append(("ALERT %s: %s%s, %s after %.0f s - review on /nurses"
                          % (name, status, by, back, down), AMBER))
        else:
            lines.append(("ALERT %s: %s%s%s" % (name, status, " %.0f s" % age if age is not None else "", by),
                          RED if status == "open" else AMBER))
    return lines


class Overlay(object):
    def __init__(self, floor_guide_y=0.65):
        try:  # Jetson-only; imported here so the text builders can be tested anywhere
            import jetson_utils as ju
        except ImportError:  # pragma: no cover
            import jetson.utils as ju
        self._ju = ju
        self._font = ju.cudaFont(size=18)
        self.floor_guide_y = floor_guide_y

    def draw(self, frame, lines, head=None, hip=None):
        """lines: [(text, colour)] top-left; head/hip: pixel points to mark, or None."""
        img, ju = frame["img"], self._ju
        try:
            if self.floor_guide_y:
                y = int(self.floor_guide_y * img.height)
                for x in range(0, img.width, 24):
                    ju.cudaDrawLine(img, (x, y), (min(x + 12, img.width - 1), y), CYAN, 2)
                self._font.OverlayText(img, text="floor zone below", x=img.width - 160, y=y + 4,
                                       color=CYAN, background=(0, 0, 0, 140))
            for point, label in ((head, "head"), (hip, "hip")):
                if point is not None:
                    px, py = int(point[0]), int(point[1])
                    ju.cudaDrawCircle(img, (px, py), 7, AMBER)
                    self._font.OverlayText(img, text=label, x=px + 9, y=py - 9, color=AMBER,
                                           background=(0, 0, 0, 140))
            y = 6
            for text, colour in lines:
                self._font.OverlayText(img, text=text, x=6, y=y, color=colour,
                                       background=(0, 0, 0, 170))
                y += 22
        except Exception:
            pass  # drawing must never stop capture
