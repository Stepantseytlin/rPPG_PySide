import numpy as np
from PySide6.QtCore import QObject, Slot

from domain.FIFO import FIFO


class FIFOWorker(QObject):
    def __init__(self, fifo: FIFO):
        super().__init__()
        self.fifo = fifo

    @Slot(np.ndarray)
    def VSlot(self, frame):
        pass

    @Slot(np.ndarray, np.ndarray)
    def InstantSlot(self, freq, pgm):
        freq = np.asarray(freq).reshape(-1)
        pgm = np.asarray(pgm)
        for num, freq_ in enumerate(freq):
            self.fifo.pIf.values.append(freq_)
            if pgm.ndim >= 2:
                self.fifo.pgm.append(pgm[num])
            else:
                self.fifo.pgm.append(pgm)
        self.fifo.sta.freqready = 1
        self.fifo.sta.wrapperoutputready = 1

    @Slot(np.ndarray)
    def PSlot(self, roi):
        self.fifo.insertFrame(roi)
        self.fifo.sta.roiinserted = max(0, self.fifo.sta.roiinserted - 1)
