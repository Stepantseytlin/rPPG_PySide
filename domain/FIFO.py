import numpy as np

from .stat import Stat
from processing.meanFrame import MeanFrameMethodDict
from src.segment_timeseries import segment_timeseries
from threadingUtils.controller import ThreadsManager
from domain.IF import IF


class FIFO(list[list[np.ndarray]]):
    def __init__(self, seglength: int = 1024, strapLen: int = 20, IF: IF = None, stat: Stat = None, config=None, UIslot=None, finslot=None):
        super().__init__()
        self.config = config
        self.sta = stat or Stat()
        if IF is None:
            self.pIf = globals()["IF"]()
        elif isinstance(IF, type):
            self.pIf = IF()
        else:
            self.pIf = IF
        self.slot = UIslot
        self.finslot = finslot
        self.frameCounter = 0
        self.segCounter = 0
        self.freqReady = 0
        self.pgm: list[np.ndarray] = []
        self.CR: list[np.ndarray] = []

        self.seglength = config.segLength if config is not None else seglength
        self.strapLen = config.strapLen if config is not None else strapLen
        source_type = config.SourceType if config is not None else "txt"
        self.MeanFrameMethod = MeanFrameMethodDict[source_type]

        self.threadmngr = ThreadsManager(1, self.config)
        self.threadmngr.make_connections(0, UIslot, finslot)
        self.append([])

    def setSlots(self, numwrk, UIslot, FinSlot):
        self.threadmngr.make_connections(numwrk, UIslot, FinSlot)

    def insertFrame(self, Frame):
        if Frame is None:
            return

        if self.frameCounter < self.seglength:
            frameRGB = self.MeanFrameMethod(Frame)
            frameRGB = np.asarray(frameRGB, dtype=np.float32)
            for segment in self:
                segment.append(frameRGB)
            self.frameCounter += 1
            if self.frameCounter % self.strapLen == 0:
                self.segCounter += 1
                self.append([])
            return

        ready = self.pop(0)
        if len(ready) == self.seglength:
            self.segCounter -= 1
            self.frameCounter = self.seglength - self.strapLen
            self.threadmngr.to_thread(ready, self.pIf, self)
            # print(f"segments ready")
        self.insertFrame(Frame)

    def resize(self,SegLength, /,Straprcnt :  None =None):
        if Straprcnt is None:
            Straprcnt = (self.config.segLength - self.config.strapLen)/self.config.segLength*100
        Straplen = None
        if Straprcnt: Straplen = int(SegLength/100 * (100-Straprcnt))
        self.config.strapLen = Straplen or self.config.strapLen
        self.config.segLength = SegLength
        observation = self.pop(0)
        self.reset()
        # self.clear()
        # self.append([])
        # self.freqReady = 0
        # self.segCounter = 0
        # self.pgm = []
        # self.CR = []
        SegmentedTimeSeries = segment_timeseries(observation,SegLength,Straprcnt/100)
        if SegmentedTimeSeries.ndim==1:
            SegmentedTimeSeries = SegmentedTimeSeries.reshape((1,-1))
        for ready in SegmentedTimeSeries[:-1,:]:
            self.threadmngr.to_thread(ready, self.pIf, self)
        self[0] = SegmentedTimeSeries[-1,:].tolist()
        k=1
        
        while len(self[k-1][self.config.strapLen:]):
            self.append(self[k-1][self.config.strapLen:])
            k+=1
        self.frameCounter = len(self[0])
        self.seglength = self.config.segLength 
        self.strapLen = self.config.strapLen 
        pass
    def reset(self):
        if hasattr(self, "threadmngr"):
            self.threadmngr.invalidate()
        self.clear()
        self.append([])
        self.frameCounter = 0
        self.freqReady = 0
        self.segCounter = 0
        self.pgm = []
        self.CR = []
        if hasattr(self.pIf, "clear"):
            self.pIf.clear()
        self.sta.clear_runtime_flags()

    def changeConfig(self, config):
        self.config = config
        self.seglength = config.segLength
        self.strapLen = config.strapLen
        self.MeanFrameMethod = MeanFrameMethodDict[config.SourceType]
        self.reset()
