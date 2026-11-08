from concurrent.futures import ProcessPoolExecutor

from PySide6.QtCore import QObject, Signal, Slot

from src.processCalc import calculate_task


class CalculateRunner(QObject):
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.pool = ProcessPoolExecutor(max_workers=1)

    @Slot(object, object)
    def submit(self, segments, config):
        future = self.pool.submit(calculate_task, segments.copy(), config)
        future.add_done_callback(self._done)

    def _done(self, future):
        try:
            freq, pgm = future.result()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(freq, pgm)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
