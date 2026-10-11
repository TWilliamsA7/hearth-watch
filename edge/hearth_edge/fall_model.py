"""The Hugging Face fall model (YOLO11n) on the Jetson's GPU with TensorRT 8.2. Off by default.

Model: Lunar255/hearth-fall-yolo11n-onnx @ 12dc1ac, the ONNX export of
melihuzunoglu/human-fall-detection @ 97261b0 (AGPL-3.0), built into a TensorRT FP16 engine by
edge/tools/get_fall_model.sh (files in <repo>/models/fall/). Measured on the Nano: 24.7 ms per
416x416 image with trtexec, ~28 ms from Python.

check(frame) -> (posture_prediction dict, ms): letterbox the camera frame (fall_decode), run the
engine, decode boxes + NMS (fall_decode), then apply Person 2's rules (fall_rules). Runs on a
background thread (face.AsyncFaceChecker) so the camera loop never waits.

GPU buffers: plain cudaMalloc + cudaMemcpy through the CUDA runtime library (ctypes), so no extra
packages. NOT managed memory: the Nano's GPU has concurrentManagedAccess = 0, so the CPU touching
a managed buffer while poseNet's kernels run kills the process (it did, in the first live test).
The copies are small (~2 MB in, ~0.1 MB out per check).
"""

import ctypes
import json
import os
import time

from hearth_edge import fall_decode, fall_rules


def default_engine_path(size=416):
    """<repo>/models/fall/best_<size>_fp16.engine (this file is <repo>/edge/hearth_edge/...)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models", "fall",
                        "best_%d_fp16.engine" % size)


class FallModel(object):
    def __init__(self, engine_path=None, size=416):
        import numpy as np
        import tensorrt as trt
        self._np = np
        self.engine_path = engine_path or default_engine_path(size)
        if not os.path.exists(self.engine_path):
            raise IOError("fall model engine not found: %s (run edge/tools/get_fall_model.sh on "
                          "the Jetson)" % self.engine_path)
        info_path = os.path.join(os.path.dirname(self.engine_path), "export_info.json")
        info = json.load(open(info_path))
        self.names = dict((int(k), v) for k, v in info["classes"].items())
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as fh, trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(fh.read())
        self._context = self._engine.create_execution_context()
        self._cudart = ctypes.CDLL("libcudart.so")
        self._ptrs, self._nbytes = [], []
        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            nbytes = int(np.prod(shape)) * 4                 # float32
            ptr = ctypes.c_void_p()
            self._check(self._cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)),
                        "cudaMalloc")
            self._ptrs.append(ptr.value)
            self._nbytes.append(nbytes)
            if self._engine.binding_is_input(i):
                self._in, self.size = i, shape[2]
            else:
                self._out, self._out_host = i, np.empty(shape, dtype=np.float32)

    @staticmethod
    def _check(code, what):
        if code != 0:
            raise RuntimeError("%s failed (CUDA error %d)" % (what, code))

    def _copy(self, dst, src, nbytes, kind):
        self._check(self._cudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                            ctypes.c_size_t(nbytes), kind), "cudaMemcpy")

    def infer_rgb(self, rgb):
        """Camera frame (RGB uint8, H x W x 3) -> detections in frame pixels."""
        x, scale, left, top = fall_decode.letterbox(rgb, self.size)
        self._copy(self._ptrs[self._in], x.ctypes.data, self._nbytes[self._in], 1)   # to GPU
        if not self._context.execute_v2(self._ptrs):
            raise RuntimeError("TensorRT execution failed")
        self._copy(self._out_host.ctypes.data, self._ptrs[self._out], self._nbytes[self._out], 2)
        return fall_decode.decode(self._out_host, self.names, scale, left, top,
                                  rgb.shape[1], rgb.shape[0],
                                  min_score=fall_rules.DETECTION_CONFIDENCE_MIN)

    def predict_rgb(self, rgb):
        """Camera frame -> (posture_prediction dict, ms) using Person 2's rules."""
        start = time.time()
        detections = self.infer_rgb(rgb)
        prediction = fall_rules.prediction_from_detections(detections, rgb.shape[1], rgb.shape[0])
        return prediction, (time.time() - start) * 1000.0

    def close(self):
        for p in self._ptrs:
            self._cudart.cudaFree(ctypes.c_void_p(p))
        self._ptrs = []


def frame_to_rgb(img):
    """Copy a camera frame (CUDA image, rgb8) into a numpy RGB array the CPU can use."""
    try:
        import jetson_utils as ju
    except ImportError:  # pragma: no cover
        import jetson.utils as ju
    ju.cudaDeviceSynchronize()
    return ju.cudaToNumpy(img)[..., :3].copy()
