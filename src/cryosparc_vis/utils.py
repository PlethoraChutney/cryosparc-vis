import numpy as np

def vmin_vmax_percentile(im:np.ndarray, percentile:float) -> tuple[np.floating, np.floating]:
    p = np.percentile(im, [percentile, 100 - percentile])
    return (np.min(p), np.max(p))