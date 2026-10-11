from threading import Event
from hearth.perception.live_io import LatestPreview


def test_slow_preview_replaces_pending_frames_without_blocking_detection():
    started, release, finished = Event(), Event(), Event()
    sent = []
    def send(value):
        sent.append(value)
        if value == 'first':
            started.set()
            assert release.wait(2)
        if value == 'newest':
            finished.set()
    preview = LatestPreview(send)
    try:
        preview.submit('first')
        assert started.wait(1)
        preview.submit('obsolete')
        preview.submit('newest')
        release.set()
        assert finished.wait(1)
        assert sent == ['first', 'newest']
    finally:
        release.set(); preview.close()
