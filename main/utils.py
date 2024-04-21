import numpy as np


def alpha_function(F, width=0.15, P=4, offset=0):
    """
    A rather heuristic function to generate the alpha values for the animation
    Can be improved A LOT

    Parameters
    :F: (int) Number of frames
    :width: (float) Width of the window (scaled to 1)
    :P: (int) Number of peaks
    :offset: (int) Offset to start the function
    """

    f_arr = np.zeros(F)
    for p in range(P):
        DT = int(width * F / P)
        t_grow = np.linspace(0, 1, DT // 2)
        t_decay = np.linspace(1, 0, DT // 2)

        idx_start = int(offset * F + p * F / P)
        idx_end1 = idx_start + DT // 2
        idx_end2 = idx_start + DT

        f_arr[idx_start:idx_end1] = t_grow
        f_arr[idx_end1:idx_end2] = t_decay

    return 0.5 * f_arr
