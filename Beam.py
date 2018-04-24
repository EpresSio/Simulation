import numpy as np
import sys
from scipy.constants import epsilon_0, pi
import math
import matplotlib.pyplot as plt


class Beam:
    # current [mA], energy [keV],r [cm]
    def __init__(self, beam_descriptor, current, energy, particle="Li_7", r=2.5):
        self.beam_descriptor = beam_descriptor
        self.r = r * 1e-2
        self.v_z, self.mass_per_q_ratio = \
            self.convert_parameters(energy * 1e3, particle)
        self.I = current * 1e-3

        self.start_charge_density_values = []
        self.charge_density_values = []
        self.end_charge_density_values = []
        self.dt = 0
        self.Q = 0

        self.start_positions = []
        self.end_positions = []

    @staticmethod
    def convert_parameters(energy, particle):
        joule_energy = energy * 1.6e-19  # J (1 eV = 1.6 * 10^-19 J)
        if particle == "Li_7":
            m = 1.16e-26  # kg [one particle mass] (atomic_mass = 6.95 g/mol, Avogadro_number = 6*10^23)
            q = 3 * 1.6e-19  # C [charge of pne particle]
            velocity = np.sqrt(2 * joule_energy / m)  # m/s (E_kin = 1/2 * m*v^2 => v^2 = 2 * E_kin/m)
        else:
            return 0, 0, 0

        mass_per_q_ratio = m / q

        return velocity, mass_per_q_ratio

    def calculate_beam(self, r_resolution, z_interval, v_r_field=None):
        # set up fields
        self.dt = float(float(z_interval) / float(self.v_z))
        self.Q = self.I * self.dt
        positions, density_values = \
            Beam.get_shape(0, self.r, r_resolution, Beam.gaussian, self.beam_descriptor.parameter_start)
        self.charge_density_values = self.Q / np.trapz(density_values, positions) * density_values
        self.start_charge_density_values = self.charge_density_values
        self.start_positions = positions

        # set up R velocity
        if v_r_field is None:
            v_r_field = np.zeros(r_resolution)

        # set up Z direction array
        start_point = self.beam_descriptor.start_points
        end_point = self.beam_descriptor.end_points
        z = np.arange(start=start_point, stop=end_point, step=z_interval)

        # do the steps
        print "Number of steps: " + str(len(z))
        for i in z:
            print "Actual position: " + str(i) + " cm"
            positions, v_r_field = self.step(z_interval, positions, v_r_field)
            # self.charge_density_values = self.Q / np.trapz(self.charge_density_values, positions)\
            #                              * self.charge_density_values

        # return with the new density point positions
        plt.show()
        self.end_charge_density_values = self.charge_density_values
        self.end_positions = positions

    def step(self, dz, r, v_r_field):
        # calculate E_r from density profile
        E = self.E_r(r)

        # calculate v_r field form density profile, E_r, and past v_r field
        result_velocity_field = self.v_r(dz, r, E, v_r_field)
        # plt.plot(r, E)
        # plt.show()
        # sys.exit()

        # calculate new density positions from past positions and v_r field
        result_r = self.q_profile(dz, r, result_velocity_field)

        # return with the new positions and new v_r field
        return result_r, result_velocity_field

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
                E_r[i] = self.integral(r[i], r) / float(epsilon_0 * r[i])
        # plt.plot(r, E_r)
        # plt.axis([0, 25, 0, 40])
        return E_r

    def integral(self, r0, r):
        E = 0
        for i in range(len(r)):
            if r[i] > 0:
                r_ = r0 - r[i]
                if math.isinf(r_):
                    # check for infinity
                    E = 0
                elif r_ > 0:
                # elif r_ != 0:
                    E = E + self.charge_density_values[i] * r[i]
        return E

    def v_r(self, dz, r, E_r, velocity_field):
        for i in range(len(r)):
            # dv_r = E*dz/(m*v_z) --> m = m_i*dV
            # F = qE
            velocity_field[i] = float(velocity_field[i]) + \
                             float(E_r[i]) * self.dt\
                                / float(self.mass_per_q_ratio * self.charge_density_values[i])
        # plt.plot(r, velocity_field)
        # plt.axis([0, 25, 0, max(velocity_field)])
        # plt.show()
        return velocity_field

    @staticmethod
    def gaussian(x, parameters):
        mu = parameters[0]
        sig = parameters[1]
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    @staticmethod
    def get_shape(start, end, resolution, function, parameters):
        positions = np.linspace(start, end, resolution)
        density_values = function(positions, parameters)
        return positions, density_values
