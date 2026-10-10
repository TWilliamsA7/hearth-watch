"""Edge settings. Every value can be overridden on the run_edge.py command line.

Perception settings only. Fall-detection thresholds live in the backend monitor, never here.
"""

SOURCE_ID = "jetson-01"
ROOM_ID = "room-01"

# Laptop backend. Over the Jetson's USB link the laptop is 192.168.55.100.
BACKEND_URL = "http://192.168.55.100:8000"
OBSERVATIONS_PATH = "/ingest/observations"
HEARTBEAT_PATH = "/ingest/heartbeat"

# Camera (Logitech C270 / Arducam, USB UVC). MJPEG keeps USB bandwidth and CPU decode low.
CAMERA = "/dev/video0"
WIDTH = 640
HEIGHT = 480
# The camera can do 30 FPS, but the pipeline uses ~10-14. Asking for 15 saves CPU time spent
# decoding frames that would be skipped (and that CPU time is what the face check runs on).
CAMERA_FPS = 15

# poseNet
POSE_MODEL = "resnet18-body"
POSE_THRESHOLD = 0.15

# poseNet reports no per-keypoint confidence (measured: Keypoint has only ID, x, y); keypoints
# below POSE_THRESHOLD are simply absent. The contract allows confidence to be omitted (null =
# unknown), so none is sent rather than a made-up number. Only set a value here if a future
# pose model really measures one.
KEYPOINT_CONFIDENCE = None

# Face detection (P1). Runs on a subset of frames; other frames send face_present = null.
FACE_MODEL = "res10"   # or "facenet"; res10 found faces in 100% of probe frames, facenet 0%
FACE_THRESHOLD = 0.5
FACE_INTERVAL_S = 0.5

# Observation rate sent to the backend (frames beyond this are processed but not sent).
OBS_HZ = 10.0
BATCH_INTERVAL_S = 0.15
HEARTBEAT_INTERVAL_S = 1.0
HTTP_TIMEOUT_S = 2.0
# About 10 s of observations are buffered while the backend is unreachable; older ones drop.
QUEUE_MAX = 100

# Video preview panel (overlay.py, only when streaming). The dashed line marks the height the
# team backend requires head and hips to be below to count as floor level
# (backend/hearth/monitor/config.json floor_head_y_min / floor_hip_y_min): an aiming aid only.
FLOOR_GUIDE_Y = 0.65
BACKEND_POLL_S = 0.5   # how often the panel asks the backend for its posture/alert view

# Hip-height smoothing for vertical_velocity, and the gap after which velocity resets.
HIP_EMA_ALPHA = 0.5
VELOCITY_RESET_GAP_S = 1.0
