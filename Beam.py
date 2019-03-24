import numpy as np

import density
from Shape import Shape

"""
Beam is a data store class which contains every information about the beam from the moment of
turning on till it's shut down.

The Beam stores the relevant information from the given BeamCalculator, and while the
calculator calculate the Beam, every steps this class's append_result() method should be called.

The calculated information like shape, background(E), and FI are stored in a matrix, which first index is the "z"
position and the second is the "r" position.
"""


class Beam:
    def __init__(self, calculator, z, start_positions):
        self.I = calculator.I
        self.energy = calculator.energy
        self.neutralization_range = calculator.neutralization_range
        self.v_z = calculator.v_z
        self.dt = calculator.dt
        self.start_point = calculator.beam_descriptor.start_points
        self.end_point = calculator.beam_descriptor.end_points
        self.r = calculator.r
        self.gas_density = calculator.gas_density

        self.z = z

        self.neutralization_index = np.inf
        if self.neutralization_range[1] != np.inf:
            for i in range(len(z)):
                if i > 0 and z[i] > self.neutralization_range[1]:
                    self.neutralization_index = i - 1
                    break

        self.result_shapes = []
        self.result_background = np.zeros(shape=(len(z), len(start_positions)))
        self.FI = np.zeros(shape=(len(z), len(start_positions)))

    """
    Append the result matrix
    """

    def append_result(self, shape, FI, background, current_index):
        self.result_shapes.append(Shape(shape.positions, shape.values, shape.Q))
        self.result_background[current_index] = background
        self.FI[current_index] = FI

    """
    Returns with the starting beam shape
    """

    def start_shape(self):
        return self.result_shapes[0].positions, self.result_shapes[0].values / self.dt * 1000

    """
    Returns with the final beam shape
    """

    def end_shape(self):
        return self.result_shapes[-1].positions, self.result_shapes[-1].values / self.dt * 1000

    """
    Returns with the final beam shape
    """

    def before_neutralization_shape(self):
        if self.neutralization_index == np.inf:
            return self.end_shape()
        return self.result_shapes[self.neutralization_index].positions, self.result_shapes[
            self.neutralization_index].values / self.dt * 1000

    """
    Returns with the difference of it's r=0 position of the beam in every z points.
    """

    def get_maximum_differences(self):
        start_maximum = max(self.result_shapes[0].values)
        maximum_differences = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            maximum_differences[i] = max(self.result_shapes[i].values) / start_maximum
        return maximum_differences

    """
    Returns with the standard deviations of the beam in every z points.
    """

    def get_std(self):
        std = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            full_positions, full_density_values = density.mirror(self.result_shapes[i].positions,
                                                                 self.result_shapes[i].values)
            full_density_values = density.normalize(full_positions, full_density_values)
            std[i] = density.std(full_positions, full_density_values)
        return std

    """
    Returns with the kurtosis of the beam in every z points.
    """

    def get_kurtosis(self):
        kurtosis = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            full_positions, full_density_values = density.mirror(self.result_shapes[i].positions,
                                                                 self.result_shapes[i].values)
            full_density_values = density.normalize(full_positions, full_density_values)
            kurtosis[i] = density.kurtosis(full_positions, full_density_values)
        return kurtosis
