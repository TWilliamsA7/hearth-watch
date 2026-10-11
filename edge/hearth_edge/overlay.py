"""Information panel on the optional video preview. DISPLAY ONLY: drawn on the stream, never sent.

    13.6 FPS | 10 obs/s | backend online                       04:58:12
    person: yes (12 keypoints) | face: yes 0.99 (0.3 s ago)
    torso 4 deg | head 0.21 | hip 0.55 | box 0.38 | drop +0.01/s
    backend: online | posture standing                 <- the backend's own view
    alerts: none open                                  <- (or: ALERT possible fall: open 4 s)
    ... head and hip dots

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


def fall_model_line(prediction, age_s):
    """The optional fall model's latest call, or None when it is off / has no result yet."""
    if prediction is None:
        return None
    label = prediction["label"]
    if label == "unknown":
        why = (prediction.get("uncertainty_reason") or "").replace("_", " ")
        return "fall model: unknown (%s, %.1f s ago)" % (why, age_s)
    return "fall model: %s %.2f (%.1f s ago)" % (label, prediction["confidence"], age_s)


def posture_mlp_line(result):
    """Our own posture model's call on this frame (posture_mlp.py)."""
    if result is None:
        return "our model: - (too few keypoints)"
    if result["label"] == "unknown":
        return "our model: unknown (%.2f)" % result["confidence"]
    return "our model: %s %.2f" % (result["label"], result["confidence"])


def posture_mlp_people_line(results):
    """Our model on every person in view, left to right -> (text, someone_on_floor).

    One person: the same as posture_mlp_line. Several: one entry each, FLOOR in capitals, and a
    warning when anyone is on the floor, whichever person the edge is tracking."""
    floor = any(r is not None and r["label"] == "floor" for r in results)
    if len(results) == 1:
        return posture_mlp_line(results[0]), floor
    parts = []
    for r in results:
        if r is None:
            parts.append("-")
        elif r["label"] == "unknown":
            parts.append("unknown")
        else:
            parts.append("%s %.2f" % (r["label"].upper() if r["label"] == "floor" else r["label"],
                                      r["confidence"]))
    text = "our model, %d people (left to right): %s" % (len(results), " | ".join(parts))
    return text + ("  <- someone on the floor" if floor else ""), floor


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


CHAR_W = 10.5   # rough width of one character of the size-18 panel font, in pixels


def wrap_text(text, width):
    """Split a panel line so it fits `width` characters: break after " | " separators or at
    spaces, never mid-word unless a word alone is too long. Continuation lines are indented."""
    indent = "   "
    lines, rest, start = [], text, 0       # start: where breaking may begin (after the indent)
    while len(rest) > width:
        cut = rest.rfind(" | ", start, width)
        if cut > start:
            cut += 2                                   # keep "|" on the first line
        else:
            cut = rest.rfind(" ", start, width + 1)
            if cut <= start:
                cut = width                            # one long word: cut it
        lines.append(rest[:cut].rstrip())
        rest, start = indent + rest[cut:].lstrip(), len(indent)
    lines.append(rest)
    return lines


class Overlay(object):
    def __init__(self):
        try:  # Jetson-only; imported here so the text builders can be tested anywhere
            import jetson_utils as ju
        except ImportError:  # pragma: no cover
            import jetson.utils as ju
        self._ju = ju
        self._font = ju.cudaFont(size=18)

    def draw(self, frame, lines, head=None, hip=None):
        """lines: [(text, colour)] top-left; head/hip: pixel points to mark, or None."""
        img, ju = frame["img"], self._ju
        try:
            for point, label in ((head, "head"), (hip, "hip")):
                if point is not None:
                    px, py = int(point[0]), int(point[1])
                    ju.cudaDrawCircle(img, (px, py), 7, AMBER)
                    self._font.OverlayText(img, text=label, x=px + 9, y=py - 9, color=AMBER,
                                           background=(0, 0, 0, 140))
            y = 6
            width_chars = max(20, int((img.width - 12) / CHAR_W))
            for text, colour in lines:
                for part in wrap_text(text, width_chars):
                    self._font.OverlayText(img, text=part, x=6, y=y, color=colour,
                                           background=(0, 0, 0, 170))
                    y += 22
        except Exception:
            pass  # drawing must never stop capture
