from multiprocessing import Process, Queue
from queue import Empty
import time

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtCore import QCoreApplication

from src.rPPGMonitorWrapper import TopLevel


def _config_to_dict(config):
    return {
        "cad": config.cad,
        "fMa": config.fMa,
        "fMi": config.fMi,
        "segLength": config.segLength,
        "strapLen": config.strapLen,
        "SourceType": config.SourceType,
        "instant": config.instant,
    }


def WebComputeProcess(config: dict, input_q: Queue, output_q: Queue):
    QCoreApplication.instance() or QCoreApplication([])
    wrapper = TopLevel(config)
    while True:
        to_insert = input_q.get()
        if to_insert is None:
            break
        result = wrapper.insert(to_insert)
        if result is None:
            output_q.put(None)
            continue
        output_q.put({
            "frequencies": result[0],
            "pgms": result[1],
            "CR": result[2],
        })


def ComputeProcess(config: dict, filepth: str, output_q: Queue):
    QCoreApplication.instance() or QCoreApplication([])
    wrapper = TopLevel(config, filepath=filepth)
    wrapper.InitRead()

    while True:
        if wrapper.fifo.sta.wrapperoutputready:
            wrapper.fifo.sta.wrapperoutputready = 0
            result = wrapper.getResults()
            output_q.put({
                "frequencies": result[0],
                "pgms": result[1],
                "CR": result[2],
            })
        time.sleep(0.03)


class WebWrapperRunner(QObject):
    finished = Signal(object, object)

    def __init__(self, config):
        super().__init__()
        self.input_q = Queue()
        self.output_q = Queue()
        self.p = Process(target=WebComputeProcess, args=(_config_to_dict(config), self.input_q, self.output_q))

    def start(self):
        if not self.p.is_alive():
            self.p.start()

    def addFrame(self, frame, timeout: float = 0.0):
        self.start()
        self.input_q.put(frame)
        try:
            return self.output_q.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        if self.p.is_alive():
            self.input_q.put(None)
            self.p.join(timeout=2)
            if self.p.is_alive():
                self.p.terminate()


class StaticWrapperRunner(QObject):
    finished = Signal(object, object)

    def __init__(self, config, filepath: str):
        super().__init__()
        self.output_q = Queue()
        self.p = Process(target=ComputeProcess, args=(_config_to_dict(config), filepath, self.output_q))

    def initRead(self):
        if not self.p.is_alive():
            self.p.start()

    def getResults(self, timeout: float | None = None):
        if timeout is None:
            return self.output_q.get()
        try:
            return self.output_q.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        if self.p.is_alive():
            self.p.terminate()
            self.p.join(timeout=2)
