import numpy as np
from .rloess import rloess
from scipy.signal import hilbert
import scipy.signal as signal
from tests.NaNsCheck import CheckNaNs
# import matplotlib.pyplot as plt
from tests.plot import plot
pi2 = np.pi*2
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.ndimage import median_filter

def suppress_light_jumps_dbscan(
    rgb: np.ndarray,
    linear_recovery: bool = True,
    eps: float = 0.002,
    min_samples: int = 10,
    scale: float = 255.0,
) -> np.ndarray:
    """
    Подавляет скачки освещения методом DBSCAN.

    linear_recovery=True:
        линейная интерполяция первых разностей — метод из статьи.

    linear_recovery=False:
        замена выбросов медианой нормальных первых разностей.
    """
    x = np.asarray(rgb, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("rgb должен иметь форму (T, 3)")
    if not np.all(np.isfinite(x)):
        raise ValueError("rgb содержит NaN или inf")
    if eps <= 0 or min_samples < 1 or scale <= 0:
        raise ValueError("Параметры должны быть положительными")
    if len(x) < 2:
        return x.copy()

    delta = np.diff(x, axis=0)
    repaired_delta = delta.copy()

    normalized = x / scale
    normalized_delta = delta / scale
    indices = np.arange(len(delta))

    for channel in range(3):
        features = np.column_stack((
            normalized[1:, channel],
            normalized_delta[:, channel],
        ))

        labels = DBSCAN(
            eps=eps,
            min_samples=min_samples,
        ).fit_predict(features)

        bad = labels == -1
        good = ~bad

        if not np.any(bad) or not np.any(good):
            continue

        if linear_recovery:
            repaired_delta[bad, channel] = np.interp(
                indices[bad],
                indices[good],
                delta[good, channel],
            )
        else:
            repaired_delta[bad, channel] = np.median(
                delta[good, channel]
            )

    repaired = np.empty_like(x)
    repaired[0] = x[0]
    repaired[1:] = x[0] + np.cumsum(repaired_delta, axis=0)

    mask = bad
    channel_masks = None

    info = {
        "mask": mask,
        "channel_masks":channel_masks,
        "recovery" : "linear" if linear_recovery else "median",

    }
    return repaired
def suppress_light_jumps(rgb: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.shape[0] < 2:
        return x.copy()

    eps = np.finfo(np.float64).eps
    log_x = np.log(np.maximum(x, eps))
    delta = np.diff(log_x, axis=0)

    common_delta = np.median(delta, axis=1)
    center = np.median(common_delta)
    scale = max(
        1.4826 * np.median(np.abs(common_delta - center)),
        eps,
    )

    bad = np.abs(common_delta - center) > threshold * scale
    good = ~bad

    if not np.any(bad) or not np.any(good):
        return x.copy()

    delta[bad] = np.median(delta[good], axis=0)

    repaired = np.empty_like(log_x)
    repaired[0] = log_x[0]
    repaired[1:] = log_x[0] + np.cumsum(delta, axis=0)

    return np.exp(repaired)
def get_PW(PPG : np.ndarray,config, lastfreq):
    assert PPG.shape[0]==config.segLength, f"Проверьте длину сегмента, получено {PPG.shape[0]} а должно быть {config.segLength}"
    # PPG = suppress_light_jumps_dbscan(PPG,)
    ## Setting default constants and accessoires from within user configurations
    cad = config.cad # Frames Per Sec
    fMi = config.fMi/60 # Minimum Frequency
    fMa = config.fMa/60 # Maximum Frequency
    filter = signal.firwin(20,[fMi,fMa],pass_zero=False,fs=cad)
    dt = 1/cad # Sampling Period, sec
    
    len_PPG = config.segLength   # количество отсчетов G-сигнала
    dfc = cad/(len_PPG-1)     # интервал дискретизации частоты, Гц
    lMax = int(len_PPG/2)        # половина от количества отсчетов сигнала
    tim = np.zeros(len_PPG)
    fa = np.zeros(len_PPG)
    SfMi = np.zeros(len_PPG)
    CfMi = np.zeros(len_PPG)
    SfMa = np.zeros(len_PPG)
    CfMa = np.zeros(len_PPG)
   
    CfMi[0] = 1.0
    CfMa[0] = 1.0
    fqc = fa[0:lMax+1]
    for i in range(1, len_PPG):
        tim[i] = tim[i-1] + dt  # время в секундах
        fa[i] = fa[i-1] + dfc   # частота в герцах
        SfMi[i] = np.sin(pi2*fMi*tim[i])
        CfMi[i] = np.cos(pi2*fMi*tim[i])
        SfMa[i] = np.sin(pi2*fMa*tim[i])
        CfMa[i] = np.cos(pi2*fMa*tim[i])
    numCH = PPG.shape[-1]
    nCH = range(numCH)#[0, 1, 2]
    trd = np.zeros((len_PPG, numCH))
    trd_wav = np.zeros((len_PPG, numCH))
    wav = np.zeros((len_PPG, numCH))

    pwc = np.zeros((len_PPG, numCH))

    SmoMax = int(2 * cad / fMi )#
    SmoMin = max(5, int(round(1  * cad / fMa)))
    # print(SmoMin,max(5, int(round(1  * cad / fMa))))
    # SmoMin = max(1*int(cad/fMa),10)#15
    # Pulse wave of the clolor channels TS
    for k in range(numCH):  # cycle upon RGB-channels
         
        # # Модель тренда цветового канала
        trd[:, nCH[k]] = rloess(PPG[:,nCH[k]], SmoMax) 
        trd_wav[:, nCH[k]]  = rloess(PPG[:,nCH[k]], SmoMin)  
        wav[:, nCH[k]] =  trd_wav[:, nCH[k]]-trd[:, nCH[k]]

        

        # Усечение спектра частотой среза fMa с помощью преобразования Гильберта
        Sc = SfMa * wav[: ,nCH[k]]
        hSc = hilbert(Sc)  # Синусовая компонента
        plot(hSc,f"debug/hSc{k}.png")
        Cc = CfMa * wav[: ,nCH[k]]
        hCc = hilbert(Cc)  # Косинусная компонента
        plot(hCc,f"debug/hCc{k}.png")
        ScroppedHigh = SfMa * np.imag(hCc) - CfMa * np.imag(hSc)
        # Экспертная модель второй информативной структурной компоненты
        expModel = ScroppedHigh
        plot(ScroppedHigh,f"debug/ScroppedHigh{k}.png")

        # # Усечение спектра частотой среза fMi с помощью преобразования Гильберта

        Sc = SfMi * expModel
        hSc = hilbert(Sc)
        Cc = CfMi * expModel
        hCc = hilbert(Cc)
        Sc = SfMi * np.imag(hCc) - CfMi * np.imag(hSc)
        CheckNaNs(Sc,"get_PW")
        # Экспертная модель 
        ScroppedLowHigh = expModel - Sc # получили быструю компоненту
        plot(ScroppedLowHigh,f"debug/ScroppedLowHigh{k}.png",do =0)
        
        wav[:, nCH[k]] = ScroppedLowHigh
        pwc[:, nCH[k]] = wav[:, nCH[k]]
        CheckNaNs(pwc[:,nCH[k]],"IN GET_PW")
    if numCH == 3: pw = (112.0 * pwc[:, 0] - 93.786 * pwc[:, 1] - 18.214 * pwc[:, 2]) / 255.0 #pwc[:,1] / 255.0#
    else: pw = pwc / 255.0
    # print(numCH, pw.shape)
    # DEV = (median_filter(pw*pw,SmoMin)).reshape(-1)#np.abs(hilbert(pw*pw))#
    # STD = np.sqrt(abs(DEV))
    DEV = ((pw*pw)).reshape(-1)#np.abs(hilbert(pw*pw))#
    STD = np.sqrt(median_filter((abs(DEV)),SmoMin*2))
    STD[np.isnan(STD)] = 1
    pw = pw.reshape((PPG.shape[0],))
    CheckNaNs(pw,"FINAL OF GWT_PW")
    pw = pw/STD
    pw[np.isnan(pw)] = 1
    prcMi = np.percentile(pw, 5)
    prcMa = np.percentile(pw, 95)
    for i in range(len(pw)):
        if pw[i] < prcMi:
            pw[i] = prcMi
        if pw[i] > prcMa:
            pw[i] = prcMa
     
    return pw.reshape([pw.shape[0],1])