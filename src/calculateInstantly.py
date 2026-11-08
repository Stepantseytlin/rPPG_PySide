from domain.config import Config
from src.separateHilbert import Hilbertdevidor
import threading
from src.spectral import *
from src.rloess import rloess
from src.separateHilbert import Hilbertdevidor
from src.compute_inst_freq import compute_inst_freq
from src.SSA import compute_SSA
import numpy as np
from tests.plot import plot

def calc_thread(segments,nCH,SmoMax,SmoMin,config,dic : dict,num):

    
    trd = rloess(segments[:,:,-1], SmoMax)
    trd_wav = rloess(segments[:,:,-1], SmoMin)
    # print(len(np.argwhere(abs(trd[0])<1)))
    trdnonsafe = trd.copy()
    trd[np.abs(trd)<1] = 1
    # print(f"changed so now {len(np.argwhere(abs(trd[0])<1))}")
    wav = (trd_wav-trdnonsafe)/trd
    # fast = Hd.getHigh(wav)
    # plot(trd[0],"debug/FastInstant.png" ,1)
    # cropped = Hd.getLow(fast)
    # plot(wav[0],"debug/CroppedInstant.png",1 )
    # wav = cropped
    DEV = rloess(wav*wav, 120)
    STD = np.sqrt(DEV)
    STD[np.isnan(STD)] = 1
    # plot(STD[0],"debug/STDInstant.png" )
    wav = wav/STD
    pwc = wav # добавили один цветовой канал

    dic[num] = pwc

def calculate(TS:np.ndarray, config : Config, Hd: Hilbertdevidor=None):
    print("starting Calc Inst")
    if Hd is None:
        Hd = Hilbertdevidor(config=config)
    segs, numCH = TS.shape
    nCH =  list(range(numCH))
    trd = np.zeros((segs, numCH))
    trd_wav = np.zeros((segs, numCH))
    wav = np.zeros((segs, numCH))
    fMa = config.fMa/60
    fMi = config.fMi/60
    
    pwc = []

    SmoMax = int(2 *config.cad /fMi)#160
    SmoMin = 11#max(2*int(config.cad/fMa),11)
    # print(f"windows {SmoMax,SmoMin, fMa, fMi}")
    # Сначала выделяем нормированную пульсовую волну через тренд rloess
        
    for k in nCH:
        segments = segment_timeseries(TS[:,nCH[k]].reshape((segs,1)),segment_length=config.segLength,alias = int(100*(config.segLength-config.strapLen)/config.segLength)/100)
        if segments.shape[0]>100:
            nth = segments.shape[0]//100 + 1
            dic = {}
            workers = [threading.Thread(target=calc_thread, args =(segments[i*100:(i+1)*100], nCH,SmoMax,SmoMin,config,dic,i)) for i in range(nth)]
            for wrk in workers:
                wrk.start()
            for num, wrk in enumerate(workers):
                # print(f"waiting for wrk {num}")
                if wrk.is_alive:
                    wrk.join()
            # print("last worker finished")
            pwc.append(np.concatenate([dic[ii] for ii in range(nth)],axis=0))
            # print(f"pwc shape now: {len(pwc), pwc[0].shape}")
        else:
            trd = rloess(segments[:,:,-1], SmoMax)
            trd_wav = rloess(segments[:,:,-1], SmoMin)
            trdnonsafe = trd.copy()
            trd[np.abs(trd)<1] = 1
            # print(f"changed so now {len(np.argwhere(abs(trd[0])<1))}")
            wav = (trd_wav-trdnonsafe)/trd
            # fast = Hd.getHigh(wav)
            # plot(trd[0],"debug/FastInstant.png" ,1)
            # cropped = Hd.getLow(fast)
            # plot(wav[0],"debug/CroppedInstant.png",1 )
            # wav = cropped
            DEV = rloess(wav*wav, 120)
            STD = np.sqrt(DEV)
            STD[np.isnan(STD)] = 1
            # plot(STD[0],"debug/STDInstant.png" )
            wav = wav/STD
            pwc.append(wav) # добавили один цветовой канал
    pwc = np.stack(pwc,axis=-1)
    # pwc = pwc.reshape((segments.shape[0],pwc.shape[1],numCH))
    # print(f'finished pwc { pwc.shape}')
    pw = (112.0 * pwc[:, :, 0] - 93.786 * pwc[:, :, 1] - 18.214 * pwc[:, :, 2]) / 255.0
    # DEV = rloess(pw*pw, 120)
    # STD = np.sqrt(DEV)
    # STD[np.isnan(STD)] = 1
    # plot(STD[0],"debug/STDInstant.png",1 )
    # pw = pw/STD

    # pw = pwc[:,1] #np.mean(pwc,axis =1)#
    prcMi = np.percentile(pw, 0.5,axis = -1)
    prcMa = np.percentile(pw, 99.5,axis = -1)
    for i in range(pw.shape[0]):
        for j in range(pw.shape[1]):
            if pw[i,j] < prcMi[i]:
                pw[i,j] = prcMi[i]
            if pw[i,j] > prcMa[i]:
                pw[i,j] = prcMa[i]

    # plot(pw[0],"debug/pwInstant.png",1 )
    # print(f"now pw is shape: {pw.shape}")
    segments_tom = segment_timeseries(pw.T, config.segLength, 0.98)
    # _,segments_tom,_ = get_Thompson_periodogramm(data = pw.T , Fs = config.cad, segment_length=config.segLength, alias=0.98, smthK=3)
    # print(f"segments_tom are {segments_tom.shape}")
    # freq = np.median(compute_inst_freq(pw.reshape([pw.shape[0],config.segLength]) ) )
    #print(f" shape of freq : {freq.shape, freq,cad//2 * int(10/freq)}")
    recoveredSSA, rec_traj = compute_SSA(segments_tom[0,:,:].T  , depth = 30, nET=2,overlap=0.9999,method = "SSA")
    #CheckNaNs(freq,"getIF freq")
    SSA =  recoveredSSA#segments_tom[0,:,:].T#
    pgm_to_est, segments_tom_est,step = get_Thompson_periodogramm(segments = SSA , Fs = config.cad, segment_length=config.segLength , alias=0.98, smthK=3)
    # print("SSA shape: ", SSA.shape, pgm_to_est.shape)
    
    freq = np.median(compute_inst_freq(SSA), axis=-1 )
    return freq, np.median(pgm_to_est, axis=-1)
