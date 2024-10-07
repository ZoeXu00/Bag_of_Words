import numpy as np


def get_random_points(I, alpha):
    height, width = I.shape[:2]
    points = np.c_[np.random.randint(0, height, alpha), np.random.randint(0, width, alpha)]
    assert(len(points) == alpha)
    return points
