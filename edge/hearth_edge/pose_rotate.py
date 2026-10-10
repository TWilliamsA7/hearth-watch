"""Map poses found in a 90-degree-rotated frame back to the original frame. Pure functions.

Why: poseNet was trained with only small rotations (about +-11 degrees), so a person lying on
the floor looks unusual to it. Rotating the frame 90 degrees makes a lying person look upright;
poseNet then finds more of the body, and the keypoints are rotated back here.

Rotations follow numpy.rot90 on an image array of width W and height H:
    k = +1  counter-clockwise: original (x, y) -> rotated (y, W - 1 - x)
    k = -1  clockwise:         original (x, y) -> rotated (H - 1 - y, x)
A pose is {"keypoints": {name: (x, y)}, "bbox": (left, top, right, bottom)} in pixels.
"""


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
