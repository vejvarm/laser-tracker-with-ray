import numpy as np
from matplotlib import pyplot as plt

from transformations import Transformations

if __name__ == '__main__':
    t = Transformations()

    c_arr = np.arange(0., np.sqrt(2), 0.1)
    dx_arr = np.arange(-1., 1., 0.1)
    dy_arr = np.arange(-1., 1., 0.1)

    for c in c_arr:
        for dx in dx_arr:
            for dy in dy_arr:
                r = (2 - c + t.punish(dx, dy))

