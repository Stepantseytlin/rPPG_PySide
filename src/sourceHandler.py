from __future__ import annotations

import time

import cv2 as cv
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from domain.config import Config
from domain.stat import Stat
from src.HaarCascadeROI import getHaarROI
from src.ROIExtractor import ROIExtractor
from src.ROIExtractor2 import ROIExtractor as ROIExtractor2
from src.calculateInstantly import calculate
from time import perf_counter

class AnyHandler:
    def __init__(self, filepath=None):
        self.filepath = None
        self.framecounter = 0
        self.parent = None
        self.setFile(filepath)
        self.Extractor=  ROIExtractor2()
    def setFile(self, filepath):
        self.filepath = filepath
        self.framecounter = 0

    def nextframe(self, ret_fps = False):
        return None, None, None

    def getROI(self, frame, method: str = "Hazar"):
        if frame is None:
             
            return None
        if method == "Haar":
            return getHaarROI(frame)
            
        else:
            if not self.Extractor.initialized:
                self.Extractor.init(fps=self.parent.config.cad)
                self.TS = 0
            self.TS += int(1000/self.parent.config.cad)
            return self.Extractor.extract(frame, self.TS, fps = self.parent.config.cad)
        

    def setParent(self, parent):
        self.parent : Handler = parent


class TXTHandler(AnyHandler):
    def __init__(self, filepath=None):
        self.fileFrames = None
        super().__init__(filepath)

    def _load_txt(self, filepath):
        try:
            return np.loadtxt(filepath, delimiter=",")
        except ValueError:
            return np.loadtxt(filepath)

    def setFile(self, filepath):
        super().setFile(filepath)
        self.fileFrames = None
        if filepath is not None:
            self.fileFrames = self._load_txt(filepath)

    def nextframe(self, ret_fps = False):
         
        if self.fileFrames is None or self.framecounter >= self.fileFrames.shape[0]:
            return None,None, None
        nextframe_ = self.fileFrames[self.framecounter]
        self.framecounter += 1
        return None, nextframe_, None

    def calcInst(self):
        if self.fileFrames is None:
            return None, None
        return calculate(self.fileFrames, self.parent.config)


class VideoHandler(AnyHandler):
    def __init__(self, filepath):
        self.cap = None
        self.after_frame = 0
        super().__init__(filepath)

    def setFile(self, filepath):
        super().setFile(filepath)
        if self.cap is not None:
            self.cap.release()
        self.cap = cv.VideoCapture(filepath) if filepath is not None else None
        self.source_fps = None
        if self.cap is not None and self.cap.isOpened():
            source_fps = float(self.cap.get(cv.CAP_PROP_FPS))
            if np.isfinite(source_fps) and source_fps > 0:
                self.source_fps = source_fps

    def nextframe(self, ret_fps = False):
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        
        else:
            fps = None
        ret, frame = self.cap.read()
        # Lref = 100.0

        # L = max(cv.mean(frame)[:3])
        # alpha = Lref / max(L, 1e-6)

        # cv.addWeighted(
        #     frame, alpha,
        #     frame, 0,
        #     0,
        #     dst=frame
        # )
        timestamp = self.cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
        if self.after_frame>0: 
            after_frame_ =perf_counter()
            spf = after_frame_ - self.after_frame 
            self.after_frame = after_frame_
            fps = 1 / spf
            # print(f"FPS = {fps}")
        else:
            self.after_frame = perf_counter()
        # self.after_frame = perf_counter()
        
        
        if not ret or frame is None:
            return None, None, None
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        roi_rect = self.getROI(frame)

        display_frame = frame.copy()
        roi = None
        if roi_rect is not None and len(roi_rect) > 3:
            x, y, w, h = map(int, roi_rect)
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            roi = frame[y:y + h, x:x + w].copy()
            cv.rectangle(display_frame, (x+30, y), (x + w-40, y + h//4), (34, 197, 94), 2)
        elif isinstance(roi_rect, tuple) and len(roi_rect) == 2:
            mean_rgb_roi, marked_frame = roi_rect
            roi = mean_rgb_roi
            if marked_frame is not None:
                display_frame = marked_frame
        sample = np.r_[roi, timestamp]  # R, G, B, time
        if ret_fps:
            return sample, display_frame, fps
        return sample, display_frame, fps


class CameraHandler(VideoHandler):
    def __init__(self, filepath=0):
        super().__init__(filepath)


Hdict = {"txt": TXTHandler, "video": VideoHandler, "cam": CameraHandler}


class Handler(QObject):
    InstantSignal = Signal(np.ndarray, np.ndarray)
    VideoSignal = Signal(np.ndarray, np.ndarray)
    ProcessingSignal = Signal(np.ndarray)
    batchStateChanged = Signal(bool)
    def __init__(self, config: Config, VS, PS, filepath=None, stat=None, IS=None):
        super().__init__()
        self.handlerFinished = 0
        self.stat: Stat = stat or Stat()
        self.config = config
        self.filepath = filepath
        self.FPS_delay = self._fps_delay()
        self.fps_s = []
        if VS:
            self.VideoSignal.connect(VS)
        if PS:
            self.ProcessingSignal.connect(PS)
        else:
            self.ProcessingSignal.connect(self._defaultProcessingSlot)
        if IS:
            self.InstantSignal.connect(IS)

        self.instance : AnyHandler = self._create_instance(filepath)
        self.ChangeConfig(config,filepath)

    def _fps_delay(self):
        return 0.001 if self.config.SourceType == "txt" else max(0.001, 1 / max(1, self.config.cad))

    def _create_instance(self, filepath=None):
        handler_cls = Hdict.get(self.config.SourceType, TXTHandler)
        instance = handler_cls(filepath)
        print(F"INIT FILE PATH {filepath}")
        instance.setParent(self)
        return instance

    def calc(self):
        if not hasattr(self.instance, "calcInst"):
            return
        freq, pgm = self.instance.calcInst()
        if freq is None or pgm is None:
            return
        self.stat.freqready = 1
        self.stat.wrapperoutputready = 1
        self.InstantSignal.emit(np.asarray(freq), np.asarray(pgm))

    @Slot()
    def read(self):

        while True:
            if self.handlerFinished:
                time.sleep(0.2)
                continue

            if self.stat.Pause:
                time.sleep(0.05)
                continue

            if self.config.instant == 1 and self.config.SourceType == "txt":
                self.batchStateChanged.emit(True)
                try:
                    self.calc()
                finally:
                    self.batchStateChanged.emit(False)
                
                self.handlerFinished = 1
                self.stat.Pause = 1
                continue
            ## считаем время на один кадр
            roi, frame, fps = self.instance.nextframe(ret_fps = True)
            if fps is not None: 
                self.fps_s.append(fps)
                if len(self.fps_s)==self.config.segLength:
                    print(f"SUGGESTED FPS = {np.mean(self.fps_s)}, current = {self.config.cad}")
                    self.fps_s = []

            ## получаем время на один кадр
            if roi is None and frame is None:
            
               
                self.handlerFinished = 1
                self.stat.Pause = 1
                self.stat.stopPlotting = 0
                continue

            if self.config.SourceType == "txt":
                if frame is not None:
                    self.ProcessingSignal.emit(np.asarray(frame))
            else:
                has_roi = (
                    roi is not None
                    and len(roi) > 0
                    and roi[0] is not None
                )

                if has_roi:
                    self.stat.roiinserted += 1
                    self.ProcessingSignal.emit(roi)

                # Кадр передаём всегда, даже если лицо не найдено.
                if frame is not None:
                    self.VideoSignal.emit(frame, roi)
                    self.stat.frameready += 1
                
                
            self.config.totalFrames += 1
            time.sleep(self.FPS_delay*0.1)

    def insertRGB(self, rgb: list[np.float32]):
        self.ProcessingSignal.emit(np.asarray(rgb))

    def insertFrame(self, frame: np.ndarray):
        roi_rect, frame = self.instance.getROI(frame)
        if roi_rect is None:
            # self.ProcessingSignal.emit(frame)
            return None, None
        
        return roi_rect, frame
        self.ProcessingSignal.emit(frame[y:y + h, x:x + w])

    @Slot(object)
    def _defaultProcessingSlot(self, roi: list[np.float32] | np.ndarray):
        pass

    @Slot(object, object)
    def ChangeConfig(self, new_config, filepath=None):
        print(f"changing config to {new_config.SourceType}")


        self.config = new_config
        if filepath is not None:
            self.filepath = filepath
        self.handlerFinished = 0
        self.stat.clear_runtime_flags()

        handler_cls = Hdict.get(self.config.SourceType, TXTHandler)
        if self.instance.__class__ is not handler_cls or filepath is not None:
            old_cap = getattr(self.instance, "cap", None)
            if old_cap is not None:
                old_cap.release()

        self.instance = self._create_instance(self.filepath)
        if self.config.SourceType == "video":
            source_fps = getattr(self.instance, "source_fps", None)
            if source_fps is not None:
                self.config.cad = source_fps

        self.FPS_delay = self._fps_delay()


