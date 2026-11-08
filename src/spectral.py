import numpy as np
from scipy.signal import hilbert, periodogram, get_window, welch
from spectrum import pmtm





def segment_timeseries(data : np.ndarray, segment_length : int, alias : float = 0.5) -> np.ndarray:
    """
    Split time series into segments of given length and percentage of cover.

    Parameters
    ----------
    data : np.ndarray
        Time series array.
    segment_length : int
        Length of one segment.
    alias : float, default 0.5
        Percentage of overlapping. Step is computed as int(segment_length * (1 - alias)).
        Must be in range [0, 1].

    Returns
    -------
    np.ndarray
        Segments array of shape (n_segments, segment_length).
    """
    n = len(data)
    segments = []
    #print("alias ",alias)
    step = int(segment_length * (1 - alias))
    if step == 0:
        step = 1
        #print(f"step size replaces with 1\n")
   # print(f"step size computed {step}")
    start = 0
    while start + segment_length <= n:
        segments.append(data[start:start + segment_length])
        start += step

    return np.array(segments)

def compute_periodogramm(data : np.ndarray, Fs : float, segment_length : int = 1024, alias : float = 0.5, window : np.ndarray | str = 'hann')->tuple[np.ndarray, np.ndarray]:
    """
    First split into segments, then compute periodogramm for each segment.

    Parameters
    ----------
    data : np.ndarray
        Time series array.
    Fs : float
        Sampling frequency.
    segment_length : float, default 1024
        Length of one segment the og data will be split into.
    alias : float, default 0.5
        Percentage of overlapping. Step is computed as int(segment_length * (1 - alias)).
        Must be in range [0, 1].

    Returns
    -------
    np.ndarray
        Periodogramm array of shape (n_segments, segment_length // 2 + 1).
    """
    segments :np.ndarray = segment_timeseries(data, segment_length, alias) #Разбиваем на сегменты
    nNsegs= segments.shape[0]
    num_ch = data.shape[1]
    pgms =np.zeros((nNsegs,segment_length//2+1,num_ch))
    for n,segment in enumerate(segments):
        for ch in range(num_ch):
            f, pgms[n,:,ch] = periodogram(segment[:,ch], fs=Fs, window=window, nfft=segment_length)
    return f, np.array(pgms), segments

def get_Thompson_periodogramm(data : np.ndarray =  None,segments : np.ndarray = None, Fs : float = 30, segment_length : int = 1024, alias : float = 0.5, smthK : int = 3 ):
    """
    First split into segments, then compute periodogramm for each segment.

    Parameters
    ----------
    data : np.ndarray
        Time series array.
    Fs : float
        Sampling frequency.
    segment_length : float, default 1024
        Length of one segment the og data will be split into.
    alias : float, default 0.5
        Percentage of overlapping. Step is computed as int(segment_length * (1 - alias)).
        Must be in range [0, 1].
    smthK : int, default 3
        Smoothing factor for the periodogramm.

    Returns
    -------
    np.ndarray
        Periodogramm array of shape (n_segments, segment_length // 2 + 1).
    """
    if segments is not None:
        nNsegs= segments.shape[0]
        if len(segments.shape)<3:
            num_ch=1
            segments = segments.reshape(nNsegs,segments.shape[1],num_ch)
        else: num_ch = segments.shape[-1]
        #print(f'num channels is {num_ch}, size is {segments.shape}')
        pgms =np.zeros((nNsegs,segment_length,num_ch))
        for n,segment in enumerate(segments):
            for ch in range(num_ch):
                SPD_cmplx, w, eigs  = pmtm(segment[:,ch], NW = smthK, k=(smthK*2 -1), NFFT=segment_length)
                pgms[n,:,ch] = np.array(np.abs(np.mean(SPD_cmplx.T*w,axis=1))**2)
        return  pgms, segments, int(segment_length * (1 - alias))
    else:
        segments : np.ndarray = segment_timeseries(data, segment_length, alias) #Разбиваем на сегменты
        nNsegs= segments.shape[0]
        num_ch = data.shape[1]
        pgms =np.zeros((nNsegs,segment_length,num_ch))
        for n,segment in enumerate(segments):
            for ch in range(num_ch):
                #print(f" new segment is {segment.shape}")
                SPD_cmplx, w, eigs  = pmtm(segment[:,ch], NW = smthK, k=(smthK*2 -1), NFFT=segment_length)
                pgms[n,:,ch] = np.array(np.abs(np.mean(SPD_cmplx.T*w,axis=1))**2)
        return  pgms, segments, int(segment_length * (1 - alias))


def periodogram2ACF(periodogramm : np.ndarray)->np.ndarray:
    """
    Convert periodogramm to ACF using inverse FFT.

    Parameters
    ----------
    periodogramm : np.ndarray
        Periodogramm array of shape (n_segments, segment_length // 2 + 1).
    Fs : float
        Sampling frequency.
    tim : np.ndarray
        Time array.

    Returns
    -------
    np.ndarray
        ACF array of shape (n_segments, segment_length // 2 + 1).
    """
    acf = np.fft.irfft(periodogramm, axis=1)
    acf = np.real(acf)
    # Нормировка
    acf = acf / acf[:, 0][:, np.newaxis]
    return acf
