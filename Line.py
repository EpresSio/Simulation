import numpy as np
from numpy.linalg import det


class Line:
    def __init__(self, p1, p2):
        self.a = (p2[1] - p1[1])
        self.b = (p2[0] - p1[0])
        self.c = -(p1[0] * p2[1] - p2[0] * p1[1])


def cross(L1, L2):
    # Creamer rule
    D = det(np.array([[L1.a, L1.b], [L2.a, L2.b]]))
    if D == 0:
        return None
    Dx = -det(np.array([[L1.c, L1.b], [L2.c, L2.b]]))
    Dy = -det(np.array([[L1.a, L1.c], [L2.a, L2.c]]))

    return [Dx / D, -Dy / D]


def get_corr_at_half(x, y):
    half = np.max(y / 2)
    shift = 0
    while y[shift] < half:
        shift = shift + 1
        if shift == len(y):
            return None
    horizontal = Line([0, half], [1, half])
    vertical = None
    for i in range(len(y) - shift):
        i = i + shift
        if y[i] < half:
            vertical = Line([x[i - 1], y[i - 1]],
                            [x[i], y[i]])
            break
    cross_ = cross(horizontal, vertical)
    return cross_


def current_density_vaule_at_R(R, density_values, positions):
    neighbour_point_indexes = [0, 0]
    for i in range(len(positions)):
        if i > 0 and positions[i] > R:
            neighbour_point_indexes = [i - 1, i]
            break

    R_line = Line([0, R], [1, R])
    profile_line = Line([density_values[neighbour_point_indexes[0]],
                         positions[neighbour_point_indexes[0]]],
                        [density_values[neighbour_point_indexes[1]],
                         positions[neighbour_point_indexes[1]]])

    cross_ = cross(R_line, profile_line)
    if cross_ is None:
        return None
    return cross_
