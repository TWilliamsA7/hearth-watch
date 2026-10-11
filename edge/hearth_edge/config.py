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
# Face templates for opted-in names (face_id.py, --face-id): SFace runs only on a face res10 found
# inside a person's head box, aligned with poseNet's eyes and nose, which must be at least this far
# apart (closer than this the face is too small to tell people apart).
FACE_ID_MIN_EYE_PX = 20.0

# Observation rate sent to the backend (frames beyond this are processed but not sent).
OBS_HZ = 10.0
BATCH_INTERVAL_S = 0.15
HEARTBEAT_INTERVAL_S = 1.0
HTTP_TIMEOUT_S = 2.0
# About 10 s of observations are buffered while the backend is unreachable; older ones drop.
QUEUE_MAX = 100

BACKEND_POLL_S = 0.5   # how often the panel asks the backend for its posture/alert view

# Optional Hugging Face fall model (fall_model.py, --fall-model): TensorRT engine from
# edge/tools/get_fall_model.sh. Checked every FALL_INTERVAL_S on a background thread; the latest
# result (up to FALL_MAX_AGE_S old) goes on EVERY observation, because the backend restarts its
# floor timer whenever observations switch between model evidence and pose evidence.
FALL_MODEL_SIZE = 416
FALL_INTERVAL_S = 0.3
FALL_MAX_AGE_S = 1.0

# Rotated poseNet passes for people lying down (pose_rotate.RotationPolicy): only when the
# upright pose looks lying or incomplete; standing costs nothing extra.
POSE_ROTATION = True

# Follow one person (features.PersonTrack): prefer the pose nearest where the person was within
# TRACK_KEEP_S, up to TRACK_MAX_DIST of the frame width away; and if they go missing for at most
# TRACK_DROPOUT_S, skip sending that frame instead of reporting "no person".
TRACK_KEEP_S = 1.0
TRACK_MAX_DIST = 0.25
TRACK_DROPOUT_S = 0.3

# Report everyone in view (hearth_edge/people.py): one observation per person per sent frame,
# each with an anonymous person_id ("p1", ...). False (or --single-person): only the followed
# person, with no person_id, as before.
MULTI_PERSON = True

# Bridge brief keypoint dropouts (features.KeypointHold): a joint missing for up to this long
# is filled in from its last position, moved with the body, for the posture numbers only.
# Observations still carry only the keypoints poseNet actually reported. 0 = off.
KEYPOINT_HOLD_S = 0.3

# Hip-height smoothing for vertical_velocity, and the gap after which velocity resets.
HIP_EMA_ALPHA = 0.5
VELOCITY_RESET_GAP_S = 1.0
