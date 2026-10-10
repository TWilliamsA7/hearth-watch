"""Contrast retry for uncertain model output, with no geometry-based posture guesses."""
from .fall_model import observation_from_result


def enhance_contrast(frame):
    import cv2
    # Enhance luminance only; preserve color and frame geometry.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def infer_posture(model, frame, *, settings, device, context, timestamp,
                  enhancement="clahe", retry_threshold=.65):
    options = dict(device=device, imgsz=settings.image_size,
                   conf=settings.detection_confidence_min, agnostic_nms=True, save=False, verbose=False)
    result = model.predict(frame, **options)[0]
    obs = observation_from_result(result, timestamp=timestamp, settings=settings, **context)
    prediction = obs.posture_prediction
    uncertain = (prediction.detection_count == 0 or
                 (prediction.detection_count == 1 and prediction.label != "unknown"
                  and prediction.confidence < retry_threshold))
    if enhancement == "clahe" and uncertain:
        alternative_result = model.predict(enhance_contrast(frame), **options)[0]
        alternative = observation_from_result(alternative_result, timestamp=timestamp, settings=settings, **context)
        candidate = alternative.posture_prediction
        # Do not upgrade unknown/multi-person output or inflate model confidence.
        if (candidate.detection_count == 1 and candidate.label != "unknown"
                and candidate.confidence >= retry_threshold):
            candidate.input_processing = "clahe"
            obs, result = alternative, alternative_result
    return obs, result


def draw_status(frame, obs, *, posture_threshold=.65, help_label="fist"):
    import cv2
    prediction = obs.posture_prediction
    reliable = (prediction.detection_count == 1 and prediction.label != "unknown"
                and prediction.confidence is not None and prediction.confidence >= posture_threshold)
    label = {"fallen":"LYING", "sitting":"SITTING", "standing":"STANDING"}.get(prediction.label, "UNKNOWN") if reliable else "UNKNOWN"
    score = f" {prediction.confidence:.0%}" if prediction.confidence is not None else ""
    color = (0, 165, 255) if label == "LYING" else (230, 230, 230)
    lines = [(f"POSTURE: {label}{score}", color),
             (f"Input: {prediction.input_processing} | Lying is posture, not proof of a fall", (230,230,230))]
    if obs.hand_gesture_prediction:
        hand = obs.hand_gesture_prediction
        lines.append((f"Hand: {hand.label} | Hold {help_label.upper()} to request help", (230,230,230)))
    scale = max(.35, min(.65, frame.shape[1] / 1000))
    for i, (text, color) in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(frame,text,(10,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    return frame
