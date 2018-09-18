import sys

import numpy as np
from scipy.constants import epsilon_0
import math

import matplotlib.pyplot as plt
from Beam import Beam


class BeamCalculator:
    # current [mA], energy [keV],r [cm], n_p [cm]
    def __init__(self, beam_descriptor, current, energy, particle="Li_7", r=2.5, neutralization_point=np.inf):
        self.beam_descriptor = beam_descriptor
        self.energy = energy
        self.r = r * 1e-2
        self.neutralization_point = neutralization_point * 1e-2
        self.v_z, self.mass_per_q_ratio = \
            self.convert_parameters(energy * 1e3, particle)
        self.I = current * 1e-3
        self.electron_factor = 0

        self.start_charge_density_values = []
        self.charge_density_values = []
        self.end_charge_density_values = []
        self.dt = 0
        self.Q = 0

        self.start_positions = []
        self.end_positions = []

        beam_parameters_string = "Beam parameter:\n" \
              "\tCurrent: " + str(current) + " mA\n" \
              "\tEnergy: " + str(energy) + " keV\n" \
              "\tStart point: " + str(beam_descriptor.start_points[0] * 1e2) + " cm\n" \
              "\tEnd point: " + str(beam_descriptor.end_points[0] * 1e2) + " cm\n"
        if neutralization_point < np.inf:
            beam_parameters_string = beam_parameters_string + "\tNeutralization point: " + str(
                neutralization_point) + " cm\n"
        # print beam_parameters_string

    @staticmethod
    def convert_parameters(energy, particle):
        joule_energy = energy * 1.6e-19  # J (1 eV = 1.6 * 10^-19 J)
        if particle == "Li_7":
            m = 1.16e-26  # kg [one particle mass] (atomic_mass = 6.95 g/mol, Avogadro_number = 6*10^23)
            q = 3 * 1.6e-19  # C [charge of one particle]
            velocity = np.sqrt(2 * joule_energy / m)  # m/s (E_kin = 1/2 * m*v^2 => v^2 = 2 * E_kin/m)
        else:
            print str(particle) + " not supported"
            return 0, 1

        mass_per_q_ratio = m / q

        return velocity, mass_per_q_ratio

    def calculate_beam(self, r_resolution, z_interval, v_r_field=None, gas_density = 0):
        z_interval = z_interval * 1e-2
        # set up fields
        self.dt = float(float(z_interval) / float(self.v_z))
        self.Q = self.I * self.dt
        positions, density_values = \
            BeamCalculator.get_shape(0, self.r, r_resolution, BeamCalculator.gaussian, self.beam_descriptor.parameter_start)

        self.charge_density_values = self.Q / np.trapz(density_values, positions) * density_values
        self.start_charge_density_values = self.charge_density_values
        self.start_positions = positions

        # set up R velocity
        if v_r_field is None:
            v_r_field = np.zeros(r_resolution)

        # set up Z direction array
        start_point = self.beam_descriptor.start_points
        end_point = self.beam_descriptor.end_points
        z = np.arange(start=start_point, stop=end_point+z_interval, step=z_interval)
        actual_z = start_point

        beam = Beam(self, z, positions)
        E = []

        # do the steps
        # print "Number of steps: " + str(len(z))
        for i in z:
            actual_z = actual_z + z_interval
            # print "Actual position: " + str(i * 1e2) + " cm"
            if actual_z < self.neutralization_point:
                positions, v_r_field, E = self.step(z_interval, positions, v_r_field)
                self.charge_density_values = self.Q / np.trapz(self.charge_density_values, positions) \
                                             * self.charge_density_values
            else:
                positions = self.step_after_neutralization(z_interval, positions, v_r_field)
                self.charge_density_values = self.Q / np.trapz(self.charge_density_values, positions) \
                                             * self.charge_density_values

            beam.append_result(positions, self.charge_density_values, self.calculateFI(E, positions))

        # return with the new density point positions
        self.end_charge_density_values = self.charge_density_values
        self.end_positions = positions
        return beam

    def calculateFI(self, E, positions):
        FI = np.zeros(shape=(len(E)))
        for i in range(len(E)):
            FI[i] = -np.trapz(E[:i], positions[:i])
        return FI

    def step(self, dz, r, v_r_field):
        # calculate E_r from density profile
        E = self.E_r(r)

        # calculate v_r field form density profile, E_r, and past v_r field
        result_velocity_field = self.v_r(r, E, v_r_field)

        # calculate new density positions from past positions and v_r field
        result_r = self.q_profile(dz, r, result_velocity_field)

        # return with the new positions and new v_r field
        return result_r, result_velocity_field, E

    def step_after_neutralization(self, dz, r, v_r_field):
        result_r = self.q_profile(dz, r, v_r_field)

        # return with the new positions
        return result_r

    def q_profile(self, dz, r, velocity_field):
        # calculate time delay form z direction velocity and step size: dt = dz/v_z
        result_r = np.zeros(shape=r.shape)
        for i in range(len(r)):
            # r(z+dz) = r(z) + v*dt
            result_r[i] = r[i] + velocity_field[i] * self.dt
        return result_r

    def E_r(self, r):
        E_r = np.zeros(shape=r.shape)
        for i in range(len(r)):
            if math.isinf(r[i]) or r[i] == 0:
                # check for infinity
                E_r[i] = 0
            else:
                E_r[i] = self.integral(r[i], r) / float(epsilon_0 * r[i]) * 1000
                # E_r[i] = np.trapz(self.charge_density_values*r, r) / float(epsilon_0 * r[i])
        return E_r

    def integral(self, r0, r):
        E = 0
        for i in range(len(r)):
            r_ = r0 - r[i]
            if math.isinf(r_):
                # check for infinity
                E = 0
            elif r_ >= 0:
                if i > 0:
                    E = E + self.charge_density_values[i] *(1+self.electron_factor) * r[i] * (r[i]-r[i-1])
                else:
                    E = self.charge_density_values[i] *(1+self.electron_factor) * r[i]
        return E

    def v_r(self, r, E_r, velocity_field):
        for i in range(len(r)):
            # dv_r = E*dz/(m*v_z) --> m = m_i*dV
            # F = qE
            velocity_field[i] = float(velocity_field[i]) + \
                                float(E_r[i]) * self.dt \
                                / self.mass_per_q_ratio
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
    def get_shape(start, end, resolution, current_function, parameters):
        positions = np.linspace(start, end, resolution)
        density_values = current_function(positions, parameters)
        return positions, density_values


