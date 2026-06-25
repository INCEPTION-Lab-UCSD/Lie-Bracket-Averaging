import math

import numpy as np


def generate_oscillator():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


print([[j for j in range(4) if j != i] for i in range(4)])
