import numpy as np

import Line
import density
from Shape import Shape


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
                    self.neutralization_index = i-1
                    break

        self.result_shapes = []
        self.result_positions = np.zeros(shape=(len(z), len(start_positions)))
        self.result_densities = np.zeros(shape=(len(z), len(start_positions)))
        self.FI = np.zeros(shape=(len(z), len(start_positions)))

    def append_result(self, shape, FI, current_index):
        self.result_shapes.append(Shape(shape.positions, shape.values, shape.Q))
        self.result_positions[current_index] = shape.positions
        self.result_densities[current_index] = shape.values
        self.FI[current_index] = FI

    def start_shape(self):
        return self.result_positions[0], self.result_densities[0]/self.dt

    def end_shape(self):
        return self.result_positions[-1], self.result_densities[-1]/self.dt

    def before_neutralization_shape(self):
        if self.neutralization_index == np.inf:
            return self.end_shape()
        return self.result_positions[self.neutralization_index], self.result_densities[self.neutralization_index]/self.dt

    def get_maximum_differences(self):
        start_maximum = max(self.result_densities[0])
        maximum_differences = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            maximum_differences[i] = max(self.result_densities[i])/start_maximum
        return maximum_differences

    def get_std(self):
        std = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            full_positions, full_density_values = density.mirror(self.result_positions[i], self.result_densities[i])
            full_density_values = density.normalize(full_positions, full_density_values)
            std[i] = density.std(full_positions, full_density_values)
        return std

    def get_kurtosis(self):
        kurtosis = np.zeros(shape=self.z.shape)
        for i in range(len(self.z)):
            full_positions, full_density_values = density.mirror(self.result_positions[i], self.result_densities[i])
            full_density_values = density.normalize(full_positions, full_density_values)
            kurtosis[i] = density.kurtosis(full_positions, full_density_values)
        return kurtosis

    # def current_value_at_half_width(self, density_values, slice_of_beam):
    #     charge = self.half_width_point_of_slice(density_values, slice_of_beam)[0]
    #     return charge / 1
    #
    # def current_value_at_half_width_x(self, density_values, slice_of_beam, x):
    #     point = self.half_width_point_of_slice(density_values, slice_of_beam)
    #     neighbour_point_indexes = [0, 0]
    #     for i in range(len(slice_of_beam)):
    #         if i > 0 and slice_of_beam[i] > point[1]:
    #             neighbour_point_indexes = [i - 1, i]
    #             break
    #
    #     half_width_x_position_line = Line.Line([point[0], point[1]*x], [0, point[1]*x])
    #     profile_line = Line.Line([density_values[neighbour_point_indexes[0]], slice_of_beam[neighbour_point_indexes[0]]],
    #                              [density_values[neighbour_point_indexes[1]], slice_of_beam[neighbour_point_indexes[1]]])
    #
    #     cross = Line.cross(half_width_x_position_line, profile_line)
    #     charge = cross[0]
    #     return charge / 1
    #
    # def half_width_current_values(self):
    #     HW_current_values = np.zeros(len(self.z))
    #     for i in range(len(self.z)):
    #         HW_current_values[i] = self.current_value_at_half_width(self.result_positions[i], self.result_positions[i])
    #     return HW_current_values
    #
    # def half_width_current_values_x(self, x):
    #     HW_current_values = np.zeros(len(self.z))
    #     for i in range(len(self.z)):
    #         HW_current_values[i] = self.current_value_at_half_width_x(self.result_positions[i], self.result_positions[i], x)
    #     return HW_current_values

    # @staticmethod
    # def half_width_point_of_slice(density_values, positions):
    #     half_value = max(density_values)/2
    #     neighbour_point_indexes = [0, 0]
    #     for i in range(len(density_values)):
    #         if i > 0 and density_values[i] < half_value:
    #             neighbour_point_indexes = [i-1, i]
    #             break
    #
    #     half_value_line = Line.Line([half_value, 0], [half_value, 1])
    #     profile_line = Line.Line([density_values[neighbour_point_indexes[0]], positions[neighbour_point_indexes[0]]],
    #                              [density_values[neighbour_point_indexes[1]], positions[neighbour_point_indexes[1]]])
    #
    #     cross = Line.cross(half_value_line, profile_line)
    #     return cross

    @staticmethod
    def current_density_vaule_at_R(R, density_values, positions):
        neighbour_point_indexes = [0, 0]
        for i in range(len(positions)):
            if i > 0 and positions[i] > R:
                neighbour_point_indexes = [i - 1, i]
                break

        R_line = Line.Line([0,R], [1,R])
        profile_line = Line.Line([density_values[neighbour_point_indexes[0]], positions[neighbour_point_indexes[0]]],
                                 [density_values[neighbour_point_indexes[1]], positions[neighbour_point_indexes[1]]])

        cross = Line.cross(R_line, profile_line)
        return cross