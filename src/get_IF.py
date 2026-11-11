from .spectral import *
from src.get_Peak_freq import get_Peak_freq
from src.compute_inst_freq import compute_inst_freq
from .SSA import compute_SSA
from tests.NaNsCheck import CheckNaNs
import matplotlib.pyplot as plt
from src.pchip_first_derivative_coefficients import pchip_first_derivative_coefficients
import numpy as np
def get_IF(pw : np.ndarray, config):
    ## Setting default constants and accessoires from within user configurations
    cad = config.cad # Frames Per Sec
    segLen = pw.shape[0]
    
    CheckNaNs(pw,"segments Tomson")
    _,segments_tom,_ = get_Thompson_periodogramm(data = pw.reshape(pw.shape[0], 1), Fs = cad, segment_length=segLen, alias=0.98, smthK=3)
    freq= np.median(compute_inst_freq(pw.reshape([1,segLen]) ) )
    recoveredSSA, _, Eigs = compute_SSA(segments_tom[:,:,0]  , depth = int(30*0 + config.cad), nET=config.nEt,overlap=0.9999,method = "SSA")
    
    SSA =recoveredSSA
    pgm_to_est, _,_ = get_Thompson_periodogramm(segments = SSA , Fs = cad, segment_length=segLen , alias=0.98, smthK=3)

    secderiv = pchip_first_derivative_coefficients(np.arange(config.segLength/cad, step = 1/config.segLength/cad)[:pw.shape[0]], pchip_first_derivative_coefficients(np.arange(config.segLength/cad, step = 1/config.segLength/cad)[:pw.shape[0]],pw[:,0])[0])[0][10:]
    APG = secderiv/np.std(secderiv)

    # Расчет частоты фазовым способом   
    # SSA_segmented = segment_timeseries(SSA.T, segment_length=segLen//4, alias = 0.5)[:,:,0]
    InstantFreqs =compute_inst_freq(SSA,cad = config.cad)#(SSA_segmented)
    freq = np.max(InstantFreqs)
    
    std = np.std(InstantFreqs)

    # Расчет частоты максимумом спектра
    # freq_peak, weight = get_Peak_freq(spectrum = pgm_to_est)
    

    if config.deriv == False: 
        return freq, np.median(pgm_to_est,axis = 0).reshape([1,pgm_to_est.shape[1]])
    else: 
        return [freq,std], np.median(pgm_to_est,axis = 0).reshape([1,pgm_to_est.shape[1]]), APG, Eigs