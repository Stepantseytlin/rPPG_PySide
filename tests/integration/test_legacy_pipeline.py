import pytest

pytest.importorskip("cv2")
pytest.importorskip("skmisc")
pytest.importorskip("spectrum")

import numpy as np
from pathlib import Path
from src import get_PW
from src.get_PW import get_PW
from src.get_IF import get_IF
import matplotlib.pyplot as plt
from tests.NaNsCheck import CheckNaNs
class cnfgtest:
    cad : int = 30
    fMa : int = 240
    fMi : int = 40
    segLength : int = 1024 
    val = 8
    def __init__(self,cad=30,fMa=240,fMi=40,segLength = 1024):
        self.cad = cad
        self.fMa = fMa
        self.fMi = fMi
        self.segLength = segLength
def change(obj):
    obj.val = 9
def test_legacy_pipeline_matchespreference():
    config : cnfgtest = cnfgtest(30,240,40)
     
    data_path = Path(__file__).resolve().parents[2] / "sources" / "txt" / "RGB" / "segment.txt"
    rgb = np.loadtxt(data_path, delimiter=" ")
    pulse_wave = get_PW(rgb[:config.segLength],config=config)
    CheckNaNs(pulse_wave,"MainTest")
    

    IFreqtest, pgm = get_IF(pulse_wave,config=config)
    # assert rgb.shape
    plt.figure()
    plt.plot(np.arange(config.cad,step = config.cad/config.segLength)[:512],pgm[0,512:])
    assert pulse_wave.shape == (config.segLength,1)
    assert rgb[:config.segLength].shape == (config.segLength,3)
    assert IFreqtest.shape == ()

    print(f"TEST COMPLETED, shape {rgb.shape}\nFrequency is {IFreqtest}\npgm shape is : {pgm.shape}")

