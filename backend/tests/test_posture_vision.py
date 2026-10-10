from types import SimpleNamespace
import pytest

from conftest import BASE
from hearth.perception.fall_model import ModelSettings
from hearth.perception.posture_vision import infer_posture, draw_status


class Tensor:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def tolist(self): return self.values


def result(label="fallen", score=.9, count=1):
    return SimpleNamespace(orig_shape=(100,200), names={0:label}, boxes=SimpleNamespace(
        xyxy=Tensor([[10,20,180,80]]*count),cls=Tensor([0]*count),conf=Tensor([score]*count)))


def run(results, enhancement="clahe"):
    pytest.importorskip('cv2')
    import numpy as np
    class Model:
        calls = 0
        def predict(self, frame, **kwargs):
            value = results[self.calls]
            self.calls += 1
            return [value]
    model = Model()
    image = np.full((100,200,3),30,dtype=np.uint8)
    obs, _ = infer_posture(model,image,settings=ModelSettings.load(),device="cpu",
        context=dict(source="test",source_type="replay",room_id="room-01",run_id="test"),
        timestamp=BASE,enhancement=enhancement)
    assert draw_status(image.copy(),obs).shape == image.shape
    return obs, model.calls


@pytest.mark.parametrize('label', ['Standing','Sitting'])
def test_confident_original_posture_is_preserved_without_retry(label):
    obs, calls = run([result(label)])
    assert calls == 1 and obs.posture_prediction.input_processing == 'original'


def test_uncertain_original_can_use_confident_enhanced_model_lying():
    obs, calls = run([result('Sitting',.4),result('Fallen',.8)])
    assert calls == 2 and obs.posture_prediction.label == 'fallen'
    assert obs.posture_prediction.confidence == .8 and obs.posture_prediction.input_processing == 'clahe'


def test_weak_retry_does_not_inflate_score_or_promote_wide_box():
    obs, _ = run([result('Sitting',.4),result('Fallen',.5)])
    assert obs.posture_prediction.label == 'sitting' and obs.posture_prediction.confidence == .4


def test_multiple_people_are_not_retried_and_can_disable_enhancement():
    obs, calls = run([result(count=2)])
    assert calls == 1 and obs.posture_prediction.label == 'unknown'
    _, calls = run([result(score=.4)],enhancement='none')
    assert calls == 1


def test_ambiguous_retry_does_not_replace_original():
    obs, _ = run([result('Sitting',.4),result('Fallen',.9,count=2)])
    assert obs.posture_prediction.label == 'sitting' and obs.posture_prediction.input_processing == 'original'
