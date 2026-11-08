import numpy as np
import matplotlib.pyplot as plt
from processing.compIF import compIF
from domain.domains import *
from domain.FIFO import FIFO
# from src.get_pw_spec import get_spec
from pathlib import Path
from src.sourceHandler import Handler
import cv2
import pandas as pd
dt = 1/30
time = np.arange(0,60*4/dt,dt)
base = np.loadtxt("stevie.txt")

# plt.plot(time[:len(base)],base)
# plt.xlabel("time, s")
# plt.grid()
artifact = []
t_artifact = [8,44,83,110]
wid = 40
for num, t in enumerate(t_artifact):
    index = t/dt
    artifact.append(base[int(max(index-wid,0)):int(min(index+wid,len(base)))] - 100)
    # plt.figure()
    # plt.plot(time[int(max(index-wid,0)):int(min(index+wid,len(base)))],artifact[num]-100)
    np.savetxt(f"./artifact_{num}.txt",artifact[num])
def load_Video_ROI_rppg(videopath):
    cfg = Config(segLength=1024,source_type="video")
    pif = IF()
    fif = FIFO(854,20,pif,config = cfg)
    print(fif.config.segLength)
    handler = Handler(cfg,None, None, videopath)
    notend, rgb, fr = handler.instance.nextframe()
    rppg =[]
    while True:
        if not notend[0]:
            break
        else:
            if rgb is not None: rppg.append(rgb)
            notend, rgb, fr = handler.instance.nextframe()
    return np.array(rppg)

def load_rppg(folder) -> list[np.ndarray]:
    signals = []
    for video in sorted(Path(folder).rglob("*.avi")):
        cap = cv2.VideoCapture(str(video))
        rgb = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            mean_bgr = frame.mean(axis=(0, 1))
            rgb.append(mean_bgr[::-1])  # BGR → RGB

        cap.release()

        if rgb:
            signals.append(np.asarray(rgb, dtype=np.float32))

    return signals
def load_rppg_txt(path):
    signals = []
    for f in sorted(Path(path).rglob("rppg_*.txt"),
                    key=lambda path: int(path.stem.rsplit("_", 1)[1])):
        signals.append(np.loadtxt(f))

    return signals
def add_bpm_diapason(dataset:pd.DataFrame):
    diapason = pd.Series([],name="bpm_diapason")
    for row in range(len(dataset)):
        if (60.0*dataset.loc[row,"hr"]<60.0):
            diapason[len(diapason)] = "low"
        elif (60. * dataset.loc[row,"hr"]<72.0):
            diapason[len(diapason)] = "moderate"
        elif (60. * dataset.loc[row,"hr"]<89.0):
            diapason[len(diapason)] = "high"
    dataset["bpm_diapason"] = diapason
    return dataset
from sklearn.utils import resample
def imitate_bootstrap(n_boot, dat : pd.DataFrame):
    n_boot = 1000
    by_subject = (
            dat.assign(
                difference_bpm=60 * (
                    dat["logmad_error"] - dat["dbscan_error"]
                )
            )
            .groupby("subject_group", as_index=False)["difference_bpm"]
            .mean()
            .dropna()
        )
    bootstrap_means = np.empty(n_boot)
    for i in range(n_boot):
            sample = resample(
                by_subject,
                replace=True,
                n_samples=len(by_subject),
                random_state=42 + i,
            )
            bootstrap_means[i] = sample["difference_bpm"].mean()
    mean_difference = by_subject["difference_bpm"].mean()
    confidence_interval = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )
    return mean_difference, confidence_interval, bootstrap_means