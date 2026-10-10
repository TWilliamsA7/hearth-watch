"""Saves observations to JSONL recordings, for tuning the backend's fall logic and for replay.

Each line is one Observation exactly as it was (or would have been) sent: keypoints and
posture numbers only, never images or video. Files go to <repo>/recordings/ by default,
which git ignores, e.g. recordings/sit-20261010-031500.jsonl.
"""

import json
import os
import re
import time


def safe_label(label):
    """Clip label -> safe file-name part (letters, digits, _ and - only)."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_-")
    return s or "clip"


def default_dir():
    """<repo>/recordings: this file is <repo>/edge/hearth_edge/recorder.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "recordings")


class Recorder(object):
    def __init__(self, label, directory=None, clock=time.time):
        directory = directory or default_dir()
        if not os.path.isdir(directory):
            os.makedirs(directory)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
        base = "%s-%s" % (safe_label(label), stamp)
        path = os.path.join(directory, base + ".jsonl")
        n = 2
        while os.path.exists(path):  # two takes in the same second
            path = os.path.join(directory, "%s-%d.jsonl" % (base, n))
            n += 1
        self.path = path
        self.count = 0
        self._fh = open(path, "w")

    def write(self, observation):
        self._fh.write(json.dumps(observation) + "\n")
        self._fh.flush()  # a crash or Ctrl+C keeps everything written so far
        self.count += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
