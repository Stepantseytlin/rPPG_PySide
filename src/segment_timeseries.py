import numpy as np
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