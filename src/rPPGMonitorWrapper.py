# this is a top-level module that connects every needed module, then configures rppgmonitorer
# it creates and connects instances of FIFO, Handler, FIFOWorker and control manager
# it shall return frame, 
"""classes imports"""
from src.sourceHandler import Handler
from domain.FIFO import FIFO
from src.FIFOWorker import FIFOWorker
from threadingUtils.controller import ThreadsManager
# Four of these imports provide for a full spectrum of 
"""/classes imports"""

# supplementary imports for datatypes
from domain.domains import *
from PySide6.QtCore import QThread, Slot, Signal, QObject

class TopLevel(QObject):
    readSignal = Signal()
    def __init__(self, CFG: Config|dict, filepath : str= None, stat : Stat = None, InstantSlot : Slot = None):
        super().__init__()
        # setting configuration
        if type(CFG)==dict:
            self.CONFIG = Config()
            self.CONFIG.SourceType = CFG["SourceType"]
            self.CONFIG.segLength = CFG["segLength"]
            self.CONFIG.strapLen = CFG["strapLen"]
            self.CONFIG.cad = CFG ["cad"]
            self.CONFIG.fMa = CFG["fMa"]
            self.CONFIG.fMi = CFG["fMi"]
            self.CONFIG.instant = CFG["instant"]
            

        else:
            self.CONFIG = CFG

        # creating pIf for collecting instantaneous frequency results
        self.pIf = IF()

        # creating FIFO instance 
        self.fifo = FIFO(IF=self.pIf,config=self.CONFIG,stat=stat, UIslot= None, finslot=None )

         
        # creating fifo worker instance
        self.FIFOwrk = FIFOWorker(self.fifo)

        # moving it to a separate thread
        self.FIFOth = QThread()
        self.FIFOwrk.moveToThread(self.FIFOth)
        self.FIFOth.start()

        # creating handler instance
        self.sourceHandler = Handler(config = self.CONFIG, VS = None, PS = self.FIFOwrk.PSlot,filepath=filepath,stat=self.fifo.sta,IS = InstantSlot or self.FIFOwrk.InstantSlot)

        # at this point our TopLevel is completely configured 
    @Slot(np.float32)
    def CalculusReadySlot(self, InstFreq: np.float32):
        pass
    def ConfigureSlot(self,Slot:Slot):
        self.fifo.setSlots(0,Slot,None)
    def insert(self,frameRGB : np.ndarray):
        if frameRGB.ndim>=2:
            self.sourceHandler.insertFrame(frameRGB)
        else:
            self.sourceHandler.insertRGB(frameRGB)
        if self.fifo.sta.wrapperoutputready:
            self.fifo.sta.wrapperoutputready = 0
            return np.array(self.pIf.values), np.array(self.fifo.pgm), np.array(self.fifo.CR)
        else:
            return None
    def InitRead(self):
        self.ReaderThread = QThread()
        self.ReaderThread.start()
        self.sourceHandler.moveToThread(self.ReaderThread)
        self.readSignal.connect(self.sourceHandler.read)
        self.readSignal.emit()
        self.fifo.sta.Pause=0
        pass
    def getResults(self):
        return np.array(self.fifo.pIf.values),np.array(self.fifo.pgm), np.array(self.fifo.CR)