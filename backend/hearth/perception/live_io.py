"""Latest-only camera and preview I/O; slow consumers cannot build a backlog."""
from datetime import datetime, timezone
from threading import Condition, Thread


class LatestCamera:
    def __init__(self, capture):
        self.capture = capture
        self.condition = Condition()
        self.latest = None
        self.sequence = 0
        self.stopped = False
        self.worker = Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        while not self.stopped:
            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.stopped = True
                else:
                    self.sequence += 1
                    self.latest = frame, datetime.now(timezone.utc), self.sequence
                self.condition.notify_all()
            if not ok:
                return

    def read(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.stopped or self.sequence > previous, timeout=5)
            if self.stopped or self.sequence <= previous:
                raise RuntimeError('Camera stopped delivering fresh frames')
            return self.latest

    def close(self):
        self.stopped = True
        self.capture.release()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=2)


class LatestPreview:
    def __init__(self, send):
        self.send = send
        self.condition = Condition()
        self.pending = None
        self.stopped = False
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, frame):
        with self.condition:
            self.pending = frame
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                if self.stopped:
                    return
                frame, self.pending = self.pending, None
            try:
                self.send(frame)
            except (OSError, ValueError):
                pass  # Preview loss must not block observation/alert ingestion.

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        self.worker.join(timeout=2)
