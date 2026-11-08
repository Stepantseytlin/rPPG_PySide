import numpy as np


def CheckNaNs(obj, CodePosition: str = "random"):
    mask = np.isnan(obj)
    if not np.any(mask):
        return
    positions = ", ".join(map(str, np.argwhere(mask)))
    print(f"Checking NaNs in {CodePosition},\nNaN in position(-s){positions}")
