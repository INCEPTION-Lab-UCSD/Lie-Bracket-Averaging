import math

import numpy as np


def generate_oscillator():
    angle = np.random.uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
x = np.array([5, 6, 9])

print(np.cross(E, x))
