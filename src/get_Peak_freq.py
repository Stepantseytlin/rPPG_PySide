import numpy as np
def calculate_width(spectrum : np.ndarray):
    weight = sum(-np.diff(np.diff(spectrum)))

    return 1-weight
def get_Peak_freq(spectrum : np.ndarray, cad : int = 30):
    freq_bin = np.argmax(spectrum)
    freq = (cad/spectrum.shape[-1])*freq_bin
    # Расчет веса как ширины пика
    weight = calculate_width(spectrum)
    return freq, weight