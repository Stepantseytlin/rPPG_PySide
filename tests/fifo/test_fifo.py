from __future__ import annotations
import pytest

pytest.importorskip("cv2")
pytest.importorskip("skmisc")
pytest.importorskip("spectrum")

import numpy as np
from pathlib import Path
from domain.FIFO import FIFO
from domain.IF import IF
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtCore import QTimer
from domain.stat import Stat
from domain.config import Config
from PySide6.QtCore import Slot
import matplotlib.pyplot as plt

import time


fs = 30          # частота дискретизации, Гц / FPS
f = 2.5        # частота синуса, Гц
duration = 60     # длительность сигнала, секунд

t = np.arange(0, duration, 1 / fs)
w = 1*(2 * np.pi * f - 0.016 * t) + 0*np.pi
x = np.sin(w*t) #+ 0.1*np.sin(2 * np.pi * (f+1) * t)
x = np.stack([np.ones_like(x), x, np.ones_like(x)], axis=1)
np.savetxt("sources\\txt\\RGB\\Xtest22.txt", x)
# x = np.loadtxt("./sources/txt/RGB/P1H1Labnet.txt", delimiter=",", usecols=(0,1,2))
print(f"X shape is {x.shape}")
class Rn:
    active = 1
class cnfgtest:
    cad : int = 30
    fMa : int = 240
    fMi : int = 40
    segLength : int = 1024
    strapLen : int = 20
    SourceType : str = "txt"
    def __init__(self,cad=30,fMa=240,fMi=40,segLength = 1024,straplen=20):
        self.cad = cad
        self.fMa = fMa
        self.fMi = fMi
        self.segLength = segLength
        self.strapLen = straplen
class finslot:
    def __init__(self):
        self.th = Rn()
    def Finslot(self):
        self.th.active = 0
        print("Slot from class object")
    

def test_fifo():
    np.savetxt("sources\\txt\\RGB\\Xtest22.txt", x)
    global duration
    @Slot()
    def on_finitaSlot():
        print("stopped from test")
        th.th.active=0
        loop.quit()
    th = finslot()
    app = QApplication.instance() or QApplication([])#app = QCoreApplication.instance() or QCoreApplication([])

    loop = QEventLoop()
    pIf = IF()
    sta = Stat()
    config = Config(segLength=1024,straplen=20)
    @Slot(np.float32)
    def fifoSlot(IF : np.float32):
        # print(f"fifo len : {len(fifo), fifo.frameCounter}")
        print(f"pIf shape : {len(pIf.values)},\n{pIf.values[-1]}")
        print(f"got IF : {IF}")
        plt.figure()
        print(np.array(fifo.CR).shape)
        plt.plot(np.arange(config.cad,step = config.cad/config.segLength)[:config.segLength//2+1],np.array(fifo.pgm)[-1,0,:config.segLength//2+1])
        plt.savefig("debug/debug_pgm.png")#plt.show()
        plt.figure()
        
        plt.plot(np.array(fifo.CR)[-1,:,0])
        plt.savefig("debug/debug_PW.png")
    fifo = FIFO(config.segLength, config.strapLen,pIf, sta,  config, fifoSlot,on_finitaSlot)
    
    assert fifo

    for _ in range(x.shape[0]):
        time.sleep(0.001)
        fifo.insertFrame(x[_])#np.random.randn(3).reshape((3,))

    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    while True:
        time.sleep(0)
        print("in a ccycle")
        if th.th.active==0:
            print(f"fifo len : {len(fifo), fifo.frameCounter}")
            print(f"pIf shape : {len(pIf.values)},\n{pIf.values[-1]}")
            plt.figure()
            estimated = np.asarray(pIf.values, dtype=float).reshape(-1)
            estimated_t = np.arange(estimated.shape[0]) * config.strapLen / config.cad + config.segLength / 2 / config.cad
            plt.plot(estimated_t, estimated, label = "Esteemated")
            assert estimated.shape[0] > 0
            plt.plot(np.arange(0,duration,1/config.cad), w/2/np.pi,linestyle="--",label = "True")
            plt.legend()
            plt.ylim(0,config.fMa/60+config.fMa/60*0.2)
            plt.savefig("debug/ResultDynamic.png")
            break
        else: print("still none")

