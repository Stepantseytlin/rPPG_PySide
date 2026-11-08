from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from processing.compIF import compute_segment, config_to_dict


class ThreadsManager(QObject):
    result = Signal(np.float32)
    finishedSignal = Signal()
    failed = Signal(str)
    _processDone = Signal(object)
    _processFailed = Signal(object)

    def __init__(self, num_workers: int = 1, config=None):
        super().__init__()
        self.config = config
        self.IsactiveThread = 0
        self.pIf = None
        self.fif = None
        self._generation = 0
        self._pending_segments = deque()#maxlen=max(1, num_workers + 1)
        self._pool = ProcessPoolExecutor(max_workers=max(1, num_workers))
        self._processDone.connect(self._handle_process_result)
        self._processFailed.connect(self._handle_process_error)

    def make_connections(self, num_wrkr: int = 0, ResSlot: Slot = None, FinSlot: Slot = None):
        if ResSlot:
            self.result.connect(ResSlot)
        if FinSlot:
            self.finishedSignal.connect(FinSlot)

    def to_thread(self, segment: np.ndarray, pifObj=None, fif=None):
        self.pIf = pifObj
        self.fif = fif
        self.config = fif.config if fif is not None else self.config
        segment = np.asarray(segment, dtype=np.float32).copy()
        task = (segment, self._generation)

        if self.IsactiveThread:
            # self._submit(*task)

            self._pending_segments.append(task)
            return

        self._submit(*task)

    def _submit(self, segment: np.ndarray, generation: int):
        self.IsactiveThread = 1
        if self.fif is not None:
            self.fif.sta.threadrunning = 1
        if len(self.pIf.values)==0:
            LastFr = 1.1
        else:
            LastFr = self.pIf.values[-1]
        future = self._pool.submit(compute_segment, segment, config_to_dict(self.config),LastFr)
        future.add_done_callback(lambda done_future: self._future_done(done_future, generation))

    def _future_done(self, future, generation: int):
        try:
            self._processDone.emit({"generation": generation, "data": future.result()})
        except Exception as exc:
            self._processFailed.emit({"generation": generation, "message": str(exc)})

    @Slot(object)
    def _handle_process_result(self, payload):
        if payload["generation"] != self._generation:
            self._finish_current()
            return

        data = payload["data"]
        if self.fif is not None and self.pIf is not None:
            frequency = np.asarray(data["frequency"])
            frequency_value = (data["frequency"].item(),data["std"].item()) if data["frequency"].shape == () else (data["frequency"],data["std"])
            if self.fif.config.plotSource == "PW" :
                toPlot = np.asarray(data["pw"]) 
            elif self.fif.config.plotSource == "rPPG": 
                toPlot=data["frame"][:,1]
            elif self.fif.config.plotSource == "deriv":
                toPlot=data["deriv"]
            elif self.fif.config.plotSource == "pgm":
                toPlot=data["pgm"][0]
             
            self.fif.CR.append(toPlot)

             
             
            self.pIf.values.append([frequency_value])
             
            
            if self.fif.pgm: self.fif.pgm[0] = (np.asarray(data["pgm"]))
            else: self.fif.pgm.append(np.asarray(data["pgm"]))
            self.fif.sta.pwready = 1
            self.fif.sta.freqready = 1
            self.fif.sta.wrapperoutputready = 1
            self.fif.sta.threadrunning = 0
            self.result.emit(np.float32(np.nanmean(frequency)))

        self._finish_current()

    @Slot(object)
    def _handle_process_error(self, payload):
        if payload["generation"] == self._generation:
            if self.fif is not None:
                self.fif.sta.threadrunning = 0
            print(f"Calculation failed: {payload['message']}")
            self.failed.emit(payload["message"])
        self._finish_current()

    def _finish_current(self):
        self.IsactiveThread = 0
        self.finishedSignal.emit()
        while self._pending_segments:
            segment, generation = self._pending_segments.popleft()
            if generation == self._generation:
                self._submit(segment, generation)
                break

    @Slot()
    def finished(self):
        self._finish_current()

    def invalidate(self):
        self._generation += 1
        self._pending_segments.clear()
        if self.fif is not None:
            self.fif.sta.threadrunning = 0

    def stop(self):
        self.invalidate()
        self._pool.shutdown(wait=False, cancel_futures=True)
