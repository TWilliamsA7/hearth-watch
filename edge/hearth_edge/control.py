"""Line commands on stdin for long-running Jetson programs driven by laptop scripts.

record_clips.sh, pose_test.sh and face_test.sh start one program on the Jetson (models load
once, camera window stays live) and send it one command per line over ssh stdin when the
person presses Enter. When the sending script ends or ssh drops, stdin reaches end of input
and the program stops on its own.
"""

import select


def read_control(stream):
    """One pending command line from stream without blocking: a line, '' at end of input
    (the sender went away), or None when nothing is waiting."""
    ready, _, _ = select.select([stream], [], [], 0)
    if not ready:
        return None
    return stream.readline()
