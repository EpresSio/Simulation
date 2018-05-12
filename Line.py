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
