import numpy as np


def make_hex(N, D=15):
    """
    Make a HERA hex

    Parameters
    ----------
    N : int
        Number of antennas per side
    D : float
        Antenna spacing

    Returns
    -------
    ants : ndarray
    antvecs : ndarray
    """
    x, y, ants = [], [], []
    ant = 0
    k = 0
    start = 0
    for i in range(2*N - 1):
        for j in range(N + k):
            x.append(j + start)
            y.append(i * np.sin(np.pi/3))
            ants.append(ant)
            ant += 1
        if i < N-1:
            k += 1
            start -= .5
        else:
            k -= 1
            start += .5
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    return ants, np.vstack([x, y, np.zeros_like(x)]).T * D


