from __future__ import print_function
import sys

import numpy as np
from scipy.constants import epsilon_0
import math

import matplotlib.pyplot as plt
from Beam import Beam
from Shape import Shape


class BeamCalculator:
    # current [mA], energy [keV],r [cm], n_p [cm]
    def __init__(self, beam_descriptor, current, energy, particle="Li_7", r=2.5, neutralization_range=[np.inf, np.inf],
                 gas_density=2.5e17, e_v_r=0):
        self.beam_descriptor = beam_descriptor
        self.energy = energy
        self.r = r * 1e-2
        self.neutralization_range = [neutralization_range[0] * 1e-2, neutralization_range[1] * 1e-2]
        self.v_z, self.mass_per_q_ratio = \
            self.convert_parameters(energy * 1e3, particle)
        self.I = current * 1e-3
        self.gas_density = gas_density
        self.sigma = 1e-16
        self.electron_factor = self.sigma * self.v_z * self.gas_density

        self.start_charge_density_values = []
        self.end_charge_density_values = []
        self.dt = 0
        self.Q = 0

        self.e_v_r = e_v_r

        self.log(beam_descriptor, current, energy, neutralization_range)

    def log(self, beam_descriptor, current, energy, neutralization_range):
        beam_parameters_string = "Beam parameters:\n" \
                                 "\tCurrent: " + str(current) + " mA\n" \
                                                                "\tEnergy: " + str(energy) + " keV\n" \
                                                                                             "\tStart point: " + str(
            beam_descriptor.start_points[0] * 1e2) + " cm\n" \
                                                     "\tEnd point: " + str(
            beam_descriptor.end_points[0] * 1e2) + " cm\n"
        if neutralization_range < np.inf:
            beam_parameters_string = beam_parameters_string + "\tNeutralization point: " + str(
                neutralization_range) + " cm\n"
        print(beam_parameters_string)

    @staticmethod
    def convert_parameters(energy, particle):
        joule_energy = energy * 1.6e-19  # J (1 eV = 1.6 * 10^-19 J)
        if particle == "Li_7":
            m = 1.16e-26  # kg [one particle mass] (atomic_mass = 6.95 g/mol, Avogadro_number = 6*10^23)
            q = 3 * 1.6e-19  # C [charge of one particle]
            velocity = np.sqrt(2 * joule_energy / m)  # m/s (E_kin = 1/2 * m*v^2 => v^2 = 2 * E_kin/m)
        else:
            print(str(particle) + " not supported")
            return 0, 1

        mass_per_q_ratio = m / q

        return velocity, mass_per_q_ratio

    def calculate_beam(self, r_resolution, z_interval, v_r_field=None):
        z_interval = z_interval * 1e-2
        # set up fields
        self.dt = float(float(z_interval) / float(self.v_z))
        self.electron_factor = self.electron_factor * self.dt
        self.Q = self.I * self.dt

        # set up shape
        shape = self.get_shape(0, self.r, r_resolution, BeamCalculator.gaussian,
                               self.beam_descriptor.parameter_start, self.Q)

        # set up R velocity
        if v_r_field is None:
            v_r_field = np.zeros(r_resolution)

        # set up Z direction array
        start_point = self.beam_descriptor.start_points[0]
        end_point = self.beam_descriptor.end_points[0]
        z = np.arange(start=start_point, stop=end_point + z_interval, step=z_interval)

        # set up neutralization logic
        neutralization_rate = (self.neutralization_range[1] - self.neutralization_range[0]) / z_interval
        neutralization_step = 0

        beam = self.calculate(shape, start_point, z, z_interval, v_r_field, neutralization_rate, neutralization_step)

        # return with the new density point positions
        return beam

    def calculate(self, input_shape, start_point, z, z_interval, v_r_field, neutralization_rate, input_neutralization_step):
        # create the beam
        # do the steps
        e_background = []
        prev_beam = None
        beams = []
        if self.e_v_r == 0:
            recalculate_number = 0
        else:
            recalculate_number = len(z)
        recalculate_number = 2
        for recal in range(recalculate_number):
            print(str(recal + 1) + " of " + str(recalculate_number) + " started.")
            shape = input_shape.copy_shape()
            beam = Beam(self, z, shape.positions)
            E = []
            actual_z = start_point
            neutralization_step = input_neutralization_step
            copy_v_r_field = np.zeros(len(v_r_field))
            for i in range(len(v_r_field)):
                copy_v_r_field[i] = v_r_field[i]
            for index in range(len(z)):
                if prev_beam is not None:
                    background_density = prev_beam.result_background[index]
                    # background_density = np.zeros(shape.positions.shape)
                else:
                    background_density = np.zeros(shape.positions.shape)
                actual_z = actual_z + z_interval
                # print("Actual position: " + str(index * 1e2) + " cm")
                if actual_z < self.neutralization_range[1]:
                    if actual_z > self.neutralization_range[0]:
                        shape.values = shape.values - neutralization_step * (shape.values / neutralization_rate)
                        neutralization_step = neutralization_step + 1
                        # background_density = np.zeros(shape.positions.shape)
                    shape.positions, copy_v_r_field, E, e_background = self.step(shape, copy_v_r_field, self.dt,
                                                                            self.mass_per_q_ratio, background_density)
                    shape.values = shape.Q / np.trapz(shape.values, shape.positions) * shape.values
                else:
                    shape.positions = self.step_after_neutralization(shape.positions, copy_v_r_field, self.dt)
                    shape.values = shape.Q / np.trapz(shape.values, shape.positions) * shape.values

                beam.append_result(shape, self.calculateFI(E, shape.positions), e_background.values +
                                   background_density, index)
                beams.append(beam)
            prev_beam = beam
        beam.result_shapes[0] = beams[0].result_shapes[0]
        return beam

    @staticmethod
    def calculateFI(E, positions):
        FI = np.zeros(shape=(len(E)))
        for i in range(len(E)):
            FI[i] = -np.trapz(E[:i], positions[:i])
        return FI

    def step(self, shape, v_r_field, dt, mass_per_q_ratio, background_density):
        r = shape.positions
        density_values = shape.values
        electron_density_values = density_values * self.electron_factor
        density_values = density_values - electron_density_values

        density_values = density_values + background_density

        # calculate E_r from density profile
        E = self.E_r(r, density_values)

        FI = self.calculateFI(E, shape.positions)

        e = self.calculate_left_e_density(shape.positions, electron_density_values, FI, self.e_v_r)

        # calculate v_r field form density profile, E_r, and past v_r field
        result_velocity_field = self.v_r(r, E, v_r_field, dt, mass_per_q_ratio)

        # calculate new density positions from past positions and v_r field
        result_r = self.q_profile(r, result_velocity_field, dt)

        # return with the new positions and new v_r field
        return result_r, result_velocity_field, E, e

    def step_after_neutralization(self, r, v_r_field, dt):
        result_r = self.q_profile(r, v_r_field, dt)

        # return with the new positions
        return result_r

    @staticmethod
    def q_profile(r, velocity_field, dt):
        # calculate time delay form z direction velocity and step size: dt = dz/v_z
        result_r = np.zeros(shape=r.shape)
        for i in range(len(r)):
            # r(z+dz) = r(z) + v*dt
            result_r[i] = r[i] + velocity_field[i] * dt
        return result_r

    def E_r(self, r, density_values):
        E_r = np.zeros(shape=r.shape)
        for i in range(len(r)):
            if math.isinf(r[i]) or r[i] == 0:
                # check for infinity
                E_r[i] = 0
            else:
                E_r[i] = self.integral(r[i], r, density_values) / float(epsilon_0 * r[i]) * 1000
                # E_r[i] = np.trapz(self.charge_density_values*r, r) / float(epsilon_0 * r[i])
        return E_r

    def integral(self, r0, r, density_values):
        E = 0
        for i in range(len(r)):
            r_ = r0 - r[i]
            if math.isinf(r_):
                # check for infinity
                E = 0
            elif r_ >= 0:
                if i > 0:
                    E = E + density_values[i] * r[i] * (r[i] - r[i - 1])
                else:
                    E = density_values[i] * r[i]
        return E

    @staticmethod
    def v_r(r, E_r, velocity_field, dt, mass_per_q_ratio):
        for i in range(len(r)):
            # dv_r = E*dz/(m*v_z) --> m = m_i*dV
            # F = qE
            velocity_field[i] = float(velocity_field[i]) + \
                                float(E_r[i]) * dt \
                                / mass_per_q_ratio
        return velocity_field

    @staticmethod
    def gaussian(x, parameters):
        mu = parameters[0]
        sig = parameters[1]
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @staticmethod
    def flat(x, parameters):
        result = np.ndarray(shape=x.shape)
        for i in range(len(x)):
            if x[i] > 0.05:
                result[i] = 0
            else:
                result[i] = parameters
        return result

    @staticmethod
    def get_shape(start, end, resolution, current_function, parameters, Q):
        positions = np.linspace(start, end, resolution)
        density_values = current_function(positions, parameters)
        density_values = Q / np.trapz(density_values, positions) * density_values
        return Shape(positions, density_values, Q)

    def calculate_left_e_density(self, positions, electron_density_values, FI, e_v_r):
        e_shape = Shape(positions, electron_density_values, np.trapz(electron_density_values, positions))
        e_q = 1.602176487e-19  # C
        e_m = 9.10938215e-31  # kg
        E_kin = np.full(positions.shape, e_v_r ** 2 * e_m * 1 / 2)
        FI = -1 * FI

        E = np.zeros(E_kin.shape)
        E_pot = np.zeros(E_kin.shape)
        for i in range(len(E_kin)):
            E_pot[i] = FI[i] * e_q
            E[i] = E_kin[i] + E_pot[i]

        for i in range(len(E)):
            if E[i] > (np.max(FI) - FI[0]) * e_q:
                E[i] = 0
                e_shape.values[i] = 0

        e_shape.Q = np.trapz(e_shape.positions, e_shape.values)

        # # plt.plot(e_shape.positions, E_pot, e_shape.positions, E_kin, e_shape.positions, E)
        # plt.plot(e_shape.positions, e_shape.values)
        # plt.show()

        v_0_r = np.zeros(E_kin.shape)
        for i in range(len(E_kin)):
            v_0_r[i] = np.sqrt(2 * E[i] / e_m)

        n = np.zeros(v_0_r.shape)
        for i in range(len(E_kin)):
            if v_0_r[i] != 0:
                n[i] = 1 / v_0_r[i]

        area = np.trapz(n, e_shape.positions)
        if area == 0:
            e_shape.values = np.zeros(e_shape.values.shape)
        else:
            e_shape.values = e_shape.Q / area * e_shape.values
        # plt.plot(e_shape.positions, n)
        # plt.show()
        # sys.exit(0)
        return e_shape
