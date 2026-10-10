"""Deliberate raised-arm requests from pose keypoints, without face identity."""
from dataclasses import dataclass
from datetime import datetime

from hearth_schema import Event, Observation

from .config import MonitorConfig
from .detector import SourceKey, make_event, source_key


@dataclass
class GestureState:
    last_timestamp: datetime | None = None
    since: datetime | None = None
    side: str | None = None
    latched: bool = False
    released_since: datetime | None = None
    last_request: datetime | None = None
    hand_center: tuple[float, float] | None = None
    evidence_mode: str | None = None


class HelpGestureDetector:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.states: dict[SourceKey, GestureState] = {}

    def invalidate(self, key: SourceKey):
        if state := self.states.get(key):
            state.since = state.released_since = None
            # Missing frames cannot re-arm an already sent request.
            if not state.latched:
                state.side = None

    def arm_states(self, obs: Observation) -> dict[str, bool | None]:
        if prediction := obs.hand_gesture_prediction:
            reliable = (prediction.model_id == self.config.hand_gesture_model_id
                        and prediction.detection_count == 1 and prediction.label != "unknown"
                        and prediction.confidence is not None
                        and prediction.confidence >= self.config.hand_gesture_confidence_min)
            return {"hand_model": prediction.label == self.config.hand_help_label if reliable else None}
        points = {p.name: p for p in obs.keypoints or []
                  if p.confidence is not None and p.confidence >= self.config.gesture_keypoint_confidence_min}
        result = {"left": None, "right": None}
        if not obs.person_detected or "nose" not in points:
            return result
        for side in result:
            wrist, shoulder = points.get(side + "_wrist"), points.get(side + "_shoulder")
            if wrist is not None and shoulder is not None:
                result[side] = wrist.y <= min(points["nose"].y, shoulder.y) - self.config.gesture_raise_margin
        return result

    def process(self, obs: Observation) -> list[Event]:
        key = source_key(obs)
        state = self.states.setdefault(key, GestureState())
        if state.last_timestamp is not None:
            gap = (obs.timestamp - state.last_timestamp).total_seconds()
            if gap <= 0:
                return []
            if gap > self.config.max_observation_gap_seconds:
                self.invalidate(key)
        state.last_timestamp = obs.timestamp
        prediction = obs.hand_gesture_prediction
        mode = prediction.model_id + "@" + prediction.model_revision if prediction else "pose"
        if state.evidence_mode != mode:
            state.since = state.released_since = None
            state.hand_center = None
            state.evidence_mode = mode
        arms = self.arm_states(obs)
        if arms.get("hand_model") is not None:
            box = prediction.bbox
            center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
            if (state.hand_center is not None and
                    max(abs(a-b) for a, b in zip(center, state.hand_center)) > self.config.hand_gesture_max_center_shift):
                state.since = state.released_since = None
                if state.latched:
                    state.hand_center = center
                    return []
            state.hand_center = center
        else:
            state.hand_center = None
        if state.latched:
            if arms.get(state.side) is False and not any(value is True for value in arms.values()):
                if state.released_since is None:
                    state.released_since = obs.timestamp
                if (obs.timestamp - state.released_since).total_seconds() >= self.config.gesture_release_seconds:
                    state.latched, state.side = False, None
                    state.since = state.released_since = None
            else:
                state.released_since = None
            return []
        raised = state.side if arms.get(state.side) is True else next((side for side, up in arms.items() if up is True), None)
        if raised is None:
            state.since, state.side = None, None
            return []
        if raised != state.side or state.since is None:
            state.side, state.since = raised, obs.timestamp
        held = (obs.timestamp - state.since).total_seconds()
        cooled = state.last_request is None or (obs.timestamp - state.last_request).total_seconds() >= self.config.gesture_cooldown_seconds
        if held < self.config.gesture_hold_seconds or not cooled:
            return []
        state.latched, state.last_request = True, obs.timestamp
        if raised == "hand_model":
            return [make_event(obs, "help_requested", method="hand_gesture", gesture=prediction.label,
                               model_id=prediction.model_id, model_revision=prediction.model_revision,
                               model_confidence=prediction.confidence, duration_seconds=held,
                               severity="warning", reason="held_hand_help_gesture")]
        return [make_event(obs, "help_requested", method="raised_arm", arm=raised,
                           duration_seconds=held, severity="warning", reason="held_help_gesture")]
