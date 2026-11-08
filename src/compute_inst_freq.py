import numpy as np
from src.pchip_first_derivative_coefficients import pchip_first_derivative_coefficients
import scipy
from scipy.signal import medfilt, medfilt2d
from scipy.signal import hilbert
from tests.NaNsCheck import CheckNaNs
# import matplotlib.pyplot as plt

def compute_inst_freq(ACF : np.ndarray, cad : int = 30 ):
    """
    Рассчет мгновенных частот на основе АКФ методом преобразования Гильберта.

    1) Расчет мгновенной фазы
    2) Расчет мгновенной частоты как производная мгновенной фазы

    Parameters
    ----------
    win : int
        Ожидаемая длина временного ряда.
        Если не совпадает с len(TS), используется фактическая длина len(TS).
    TS : array-like
        Одномерный временной ряд.
    method : str, default='pchip'
        Метод интерполяции: 'pchip', 'cubic', 'v5cubic', 'makima', 'spline'.

    Returns
    -------
    np.ndarray
        Массив огибающей длины len(TS) (или win после выравнивания).

    Raises
    ------
    ValueError
        Если передан неизвестный method.
    """
   # instantenous_frequency = []
    #for acf in ACF:
    # 1) Преобразование Гильберта для получения аналитического сигнала
    
    CheckNaNs(ACF[0],"Hilbert")

    analytic_signal = hilbert(ACF, axis = -1)
    # 2) Извлечение мгновенной фазы
    phi = np.unwrap(np.angle(analytic_signal), axis=-1)/2
    
    f_inst = np.zeros_like(phi[:])
    
    dt = np.arange(phi.shape[-1]).reshape((ACF.shape[-1],))

    #CheckNaNs(phi[0,:1025],"compute_inst_freq")
    dphi,_,_ = pchip_first_derivative_coefficients(dt,phi[0:ACF.shape[0],:1025])#np.diff(phi, axis=1)
    dphi = np.clip(dphi, 0, None)   # если хотите неотрицательность
    
    fr = (cad/(2*np.pi)) * dphi
    Lag = int(np.floor(fr.shape[-1]*0.2*0.8))

    medfilt = medfilt2d(fr,(1,int(2*Lag + 1)))
    smoothed = medfilt

        
    if Lag < 1:
        Lag = 1
    Len = ACF.shape[-1]
    
    for s in range(ACF.shape[0]):
        # print(f"{Lag, smoothed[s,0], smoothed[s,Lag],smoothed[s,Lag-3]}")
        for i in np.arange(Lag):
            smoothed[s,i] = smoothed[s,Lag];# % левый сегмен оценки частоты
            smoothed[s,Len-1-i] = smoothed[s,Len-Lag-1]
        f_inst[s,:] = 2*smoothed[s,:]
    # print(f"pw shape in freq : {Lag, np.median(f_inst,axis=0).shape}")
    return np.abs(f_inst[:])