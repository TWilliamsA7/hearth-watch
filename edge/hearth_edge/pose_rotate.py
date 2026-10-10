"""Map poses found in a 90-degree-rotated frame back to the original frame. Pure functions.

Why: poseNet was trained with only small rotations (about +-11 degrees), so a person lying on
the floor looks unusual to it. Rotating the frame 90 degrees makes a lying person look upright;
poseNet then finds more of the body, and the keypoints are rotated back here.

Rotations follow numpy.rot90 on an image array of width W and height H:
    k = +1  counter-clockwise: original (x, y) -> rotated (y, W - 1 - x)
    k = -1  clockwise:         original (x, y) -> rotated (H - 1 - y, x)
A pose is {"keypoints": {name: (x, y)}, "bbox": (left, top, right, bottom)} in pixels.
"""


from hearth_edge import features

# Measured 2026-10-10 (edge/BENCHMARKS.md): lying across the picture, plain poseNet found
# head+shoulders+hips in 0% of frames; the frame rotated 90 degrees counter-clockwise found
# them in 100% (13 keypoints, torso 94 degrees). Standing needs no rotation.
LYING_TORSO_DEG = 45.0     # a tilted torso suggests lying: try a rotated pass
MIN_ROTATED_KEYPOINTS = 8  # rotated passes in an empty room invented 2-3 keypoint "people"
EMPTY_RETRY_S = 1.0        # with nobody detected: one direction per try, once a second
KEEP_TRYING_S = 1.0        # ...but every frame for a while after a rotated pose was accepted
UPRIGHT_EVERY = 3          # while someone stays down, run the upright pass on 1 frame in 3


def looks_lying_or_incomplete(pose):
    """True when a rotated pass might find more of the body than the upright one did."""
    if pose is None:
        return True
    kps = pose["keypoints"]
    if not features.torso_visible(kps):
        return True
    angle = features.torso_angle_deg(kps)
    if angle is not None and angle >= LYING_TORSO_DEG:
        return True
    aspect = features.bbox_aspect(pose["bbox"])
    return aspect is not None and aspect >= 1.0


def preferred_direction(pose):
    """Head right of the neck/shoulders -> rotate counter-clockwise (+1); left -> clockwise (-1).
    That turns the head to the top, so the person looks upright. None if unknown."""
    if pose is None:
        return None
    head = features.head_point(pose["keypoints"])
    neck = features.neck_point(pose["keypoints"])
    if neck is None:
        shoulders = [pose["keypoints"][n] for n in ("left_shoulder", "right_shoulder")
                     if n in pose["keypoints"]]
        neck = shoulders[0] if shoulders else None
    if head is None or neck is None or head[0] == neck[0]:
        return None
    return 1 if head[0] > neck[0] else -1


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def better_rotated(as_is, rotated, min_keypoints=MIN_ROTATED_KEYPOINTS):
    """Use the rotated pose only if it clearly found more of the same person, with a torso."""
    if rotated is None:
        return False
    kps = rotated["keypoints"]
    if len(kps) < min_keypoints or not features.torso_visible(kps):
        return False
    if as_is is not None:
        if len(kps) <= len(as_is["keypoints"]):
            return False
        if not boxes_overlap(as_is["bbox"], rotated["bbox"]):
            return False
    return True


class RotationPolicy(object):
    """Decides when to spend an extra poseNet pass on a rotated frame, and in which direction.

    Standing people cost nothing extra. A partly detected or lying person gets a rotated pass
    every frame (head side first). With nobody detected, one direction is tried per
    EMPTY_RETRY_S, alternating (each pass stalls the loop ~95 ms on the Nano), or every frame
    for KEEP_TRYING_S after a rotated pose was last accepted.
    """

    def __init__(self):
        self.last_k = 1
        self.last_accept_t = None
        self.last_empty_try_t = None
        self._empty_k = 1
        self._lying_frames = 0

    def directions(self, as_is, t):
        """Rotation directions to try this frame, in order (possibly none)."""
        if not looks_lying_or_incomplete(as_is):
            return []
        if as_is is None:
            recent = self.last_accept_t is not None and t - self.last_accept_t <= KEEP_TRYING_S
            if not recent:
                if self.last_empty_try_t is not None and t - self.last_empty_try_t < EMPTY_RETRY_S:
                    return []
                self.last_empty_try_t = t
                self._empty_k = -self._empty_k
                return [self._empty_k]
        first = preferred_direction(as_is) or self.last_k
        return [first, -first]

    def accepted(self, k, t):
        self.last_k = k
        self.last_accept_t = t

    def skip_upright(self, t):
        """While a rotated pose keeps being accepted (someone lying down), skip the upright
        poseNet pass on most frames to save time. It still runs every UPRIGHT_EVERY frames, and
        always once the rotated pose stops being accepted, so standing up is noticed quickly."""
        if self.last_accept_t is None or t - self.last_accept_t > 0.5:
            self._lying_frames = 0
            return False
        self._lying_frames += 1
        return self._lying_frames % UPRIGHT_EVERY != 0


def scale_pose(pose, factor):
    """Multiply every coordinate of a pose (keypoints and box) by factor."""
    kps = dict((n, (x * factor, y * factor)) for n, (x, y) in pose["keypoints"].items())
    l, t, r, b = pose["bbox"]
    return {"keypoints": kps, "bbox": (l * factor, t * factor, r * factor, b * factor)}


def rotate_point(x, y, k, width, height):
    """Original-frame point -> where it lands in the rotated frame."""
    if k == 1:
        return (y, width - 1 - x)
    if k == -1:
        return (height - 1 - y, x)
    raise ValueError("k must be 1 or -1")


def unrotate_point(xr, yr, k, width, height):
    """Rotated-frame point -> original-frame point. width/height are of the ORIGINAL frame."""
    if k == 1:
        return (width - 1 - yr, xr)
    if k == -1:
        return (yr, height - 1 - xr)
    raise ValueError("k must be 1 or -1")


def unrotate_pose(pose, k, width, height):
    """A pose found in the rotated frame, expressed in the original frame."""
    kps = dict((name, unrotate_point(x, y, k, width, height))
               for name, (x, y) in pose["keypoints"].items())
    left, top, right, bottom = pose["bbox"]
    corners = [unrotate_point(x, y, k, width, height)
               for x, y in ((left, top), (right, top), (left, bottom), (right, bottom))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return {"keypoints": kps, "bbox": (min(xs), min(ys), max(xs), max(ys))}
