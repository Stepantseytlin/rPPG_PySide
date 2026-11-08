import numpy as np
from scipy.signal import hilbert
from domain.config import Config
class Hilbertdevidor:
    def __init__(self,config : Config):
        self.fMa = config.fMa/60
        self.fMi = config.fMi/60
        self.cad = config.cad
        dt = 1/self.cad # Sampling Period, sec
        len_PPG = config.segLength
        self.len_PPG = config.segLength   # количество отсчетов G-сигнала
        self.dfc = self.cad/(len_PPG-1)     # интервал дискретизации частоты, Гц
        self.lMax = int(len_PPG/2)        # половина от количества отсчетов сигнала
        self.tim = np.zeros(len_PPG)
        self.fa = np.zeros(len_PPG)
        self.SfMi = np.zeros(len_PPG)
        self.CfMi = np.zeros(len_PPG)
        self.SfMa = np.zeros(len_PPG)
        self.CfMa = np.zeros(len_PPG)
        self.SfMa1 = np.zeros(len_PPG)
        self.CfMa1 = np.zeros(len_PPG)
        self.CfMi[0] = 1.0
        self.CfMa[0] = 1.0
        
        for i in range(1, len_PPG):
            self.tim[i] = self.tim[i-1] + dt  # время в секундах
            self.fa[i] = self.fa[i-1] + self.dfc   # частота в герцах
            self.SfMi[i] = np.sin(np.pi*2*self.fMi*self.tim[i])
            self.CfMi[i] = np.cos(np.pi*2*self.fMi*self.tim[i])
            self.SfMa[i] = np.sin(np.pi*2*self.fMa*self.tim[i])
            self.CfMa[i] = np.cos(np.pi*2*self.fMa*self.tim[i])
    def getHigh(self,wav:np.ndarray):
        segs,chs = wav.shape
        chs
        HighModel = []
        
        Sc = self.SfMa * wav[:,: ]
        hSc = hilbert(Sc,axis=-1)  # Синусовая компонента
        
        Cc = self.CfMa * wav[: ,:]
        hCc = hilbert(Cc,axis=-1)  # Косинусная компонента

        ScroppedHigh = self.SfMa * np.imag(hCc) - self.CfMa * np.imag(hSc)

        # Экспертная модель информативной структурной компоненты
        HighModel.append(ScroppedHigh) 
            
        return np.array(HighModel).reshape([segs,chs])
    def getLow(self, wav:np.ndarray):
        segs,chs = wav.shape
        numCH = chs
        nCH = list(range(numCH))
        LowModel = []
        
        Sc = self.SfMi * wav[: ,:]
        hSc = hilbert(Sc,axis=-1)  # Синусовая компонента
        
        Cc = self.CfMi * wav[: , :]
        hCc = hilbert(Cc,axis=-1)  # Косинусная компонента

        ScroppedLow = self.SfMi * np.imag(hCc) - self.CfMi * np.imag(hSc)

        # Экспертная модель информативной структурной компоненты
        LowModel.append(wav - ScroppedLow)
            
        return np.array(LowModel).reshape([segs,chs])


