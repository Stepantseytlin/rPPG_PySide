import pytest

pytest.importorskip("cv2")
pytest.importorskip("skmisc")
pytest.importorskip("spectrum")

from src.sourceHandler import Handler
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtCore import QTimer, QObject, QThread
from domain.stat import Stat
from PySide6.QtCore import Slot, Signal
from domain.FIFO import FIFO
from domain.IF import IF
from domain.stat import Stat
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt

@Slot(np.float32)
def result(freq : np.float32):
    pass
   
@Slot()
def finished():
    pass
     

if __name__ == "__main__":
    class cfg:
        cad : int = 30
        fMa : int = 240
        fMi : int = 40
        segLength : int = 1024
        strapLen : int = 20
        SourceType : str = "txt"
        strapLen =20

    ## GLOBAL VARS
    pIf = IF()
    sta = Stat()
    conf = cfg()
     
    fifo = FIFO(conf.segLength, conf.strapLen,pIf, sta,  conf, result,finished)
   
    class tobj(QObject):
        startSig : Signal = Signal()



    @Slot(np.ndarray)
    def VS(frame : np.ndarray):
        
        pass
            

    @Slot(np.ndarray,np.ndarray)
    def PS(frame : np.ndarray, roi : np.ndarray):
        plt.figure()
        plt.imshow(roi)
        

        plt.savefig("debug/face.png")
        fifo.insertFrame(roi)
        pass
    def test_handler():  
        
    
        app = QApplication.instance() or QApplication([])#app = QCoreApplication.instance() or QCoreApplication([])
        
        
        tester = tobj()
        
        source_path = Path(__file__).resolve().parents[2] / "sources" / "txt" / "RGB" / "P1H1Labnet.txt"
        handler = Handler(conf, VS, PS, filepath=str(source_path))
        activeThread : QThread = QThread()
        handler.moveToThread(activeThread)
        tester.startSig.connect(handler.read)
        activeThread.start()
         
        tester.startSig.emit()
         
       
        while True:
            time.sleep(0.3)
        



    # test_handler()

        
