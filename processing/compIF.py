import numpy as np

from domain.config import Config
from domain.IF import IF
from src.get_IF import get_IF
from src.get_PW import get_PW
from tests.plot import plot
from src.pchip_first_derivative_coefficients import pchip_first_derivative_coefficients

def config_to_dict(config):
    return {
        "cad": config.cad,
        "fMa": config.fMa,
        "fMi": config.fMi,
        "segLength": config.segLength,
        "strapLen": config.strapLen,
        "SourceType": config.SourceType,
        "instant": config.instant,
    }


def config_from_dict(config_dict):
    config = Config(
        cad=config_dict["cad"],
        fMa=config_dict["fMa"],
        fMi=config_dict["fMi"],
        segLength=config_dict["segLength"],
        straplen=config_dict["strapLen"],
        instant=config_dict.get("instant", 0),
        source_type=config_dict.get("SourceType", "txt"),
    )
    return config


def compute_segment(frame, config_dict, lastfreq =0):
    config = config_from_dict(config_dict)
    rgb = frame[:, :3]
    
    if config.SourceType != "txt":
        t = frame[:, 3]

        t_uniform = np.linspace(t[0], t[-1], len(t))
        print("===========interpolating===========")
        rgb_uniform = np.column_stack([
            np.interp(t_uniform, t, rgb[:, channel])
            for channel in range(3)
        ])

        config.cad = (len(t) - 1) / (t[-1] - t[0])
    else:
        rgb_uniform = rgb

    pw = get_PW(rgb_uniform, config, lastfreq)
    # pw = get_PW(np.asarray(frame), config,lastfreq)
    inst_freq, pgm,deriv = get_IF(pw, config)
     
    return {
        "frequency": np.asarray(inst_freq[0]),
        "std" : np.asarray(inst_freq[1]),
        "pgm": np.asarray(pgm),
        "pw": np.asarray(pw),
        "frame" : np.asarray(rgb_uniform),
        "deriv" : np.asarray(deriv)
    }


def compIF(frame, pIf: IF, fif, signal):
    try:
        result = compute_segment(frame, config_to_dict(fif.config))
    except Exception as exc:
        fif.sta.threadrunning = 0
        print(f"Calculation failed: {exc}")
        return

    fif.sta.pwready = 1
    toPlot = result["pw"] if fif.config.plotSource == "PW" else frame[:,:,1]
     
    if len(fif.CR)>0:
        fif.CR[-1](toPlot)
    else:
        fif.CR.append(toPlot)
     
    pIf.values.append((result["frequency"].item(),result["std"].item()) if result["frequency"].shape == () else (result["frequency"],result["std"]))
    fif.pgm.append(result["pgm"])
    fif.sta.freqready = 1
    fif.sta.wrapperoutputready = 1
    if signal is not None:signal.emit(np.float32(np.nanmean(result["frequency"])))
    # 
