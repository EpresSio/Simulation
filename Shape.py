import numpy as np


class Shape:
    def __init__(self, positions, values, Q):
        if len(positions) != len(values):
            raise Exception('Positions and values len should be the same')
        self.Q = Q
        self.positions = positions
        self.values = values
        self.len = len(positions)

    def copy_shape(self):
        copy_positions = np.zeros(self.len)
        copy_values = np.zeros(self.len)
        for i in range(self.len):
            copy_positions[i] = self.positions[i]
            copy_values[i] = self.values[i]
        copy_shape = Shape(copy_positions, copy_values, self.Q)
        return copy_shape
