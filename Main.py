# coding=utf-8
from __future__ import unicode_literals

from BeamCalculator import BeamCalculator
from BeamDescriptor import BeamDescription

import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.constants


def plot_beam(beam):
    p = re.compile(r'([-+]?\d*\.\d+|\d+)')
    m = p.findall(str(beam.gas_density))

    if len(m) == 2:
        gas_density_string = "$" + str(m[0][:4]) + " \cdot 10^{" + str(m[1]) + "}$"
    else:
        gas_density_string = str(beam.gas_density)
    neutralization_title = ", \nNeutralizációs tartomány = " + str(beam.neutralization_range[0]) + "-" + str(
        beam.neutralization_range[1]) + " m"
    is_neutralized = beam.neutralization_range[0] != np.inf
    if not is_neutralized:
        neutralization_title = ""
    title = "I = " + str(beam.I * 2 * 1e3) + " mA, E = " + str(
        beam.energy) + " keV, Gáz sűrűség = " + gas_density_string + " $m^{-2}$\n" + "Kezdő pont = " + str(
        beam.start_point[0]) + " m, Vég pont = " + str(
        beam.end_point[0]) + " m" + neutralization_title
    plt.title(title)

    plt.plot(beam.start_shape()[0], beam.start_shape()[1], "k", label="Kezdeti nyaláb alak")
    # if is_neutralized:
    #     plt.plot(beam.before_neutralization_shape()[0], beam.before_neutralization_shape()[1], 'y--',
    #              label="Nyaláb alak a neutralizáció után")
    plt.plot(beam.end_shape()[0], beam.end_shape()[1], 'r', label="Végső nyaláb alak")

    q_min = min(beam.start_shape()[1])
    q_max = max(beam.start_shape()[1])
    r_min = 0
    r_max = beam.r

    plt.ylabel("Felületi áram sűrűség [$\\frac{mA}{m^2}$]", fontsize=18)
    plt.xlabel("Radiális pozíció [m]", fontsize=18)
    plt.axis([r_min, r_max, q_min, q_max])
    plt.legend()
    plt.show()


def plot_FI(beam):
    plt.title("Potenciál \nI = " + str(beam.I * 1e3) + " mA, E = " + str(beam.energy) + " keV\n"
              + "Kezdő pont = " + str(beam.start_point[0]) + " m, Vég pont = " + str(beam.end_point[0]) + " m, \n"
              + "Neutralizációs tartomány = " + str(beam.neutralization_range[0]) + "-" + str(
        beam.neutralization_range[1]) + " m")

    plt.plot(beam.result_positions[0], beam.FI[0], "k", label="Initial potential")
    plt.plot(beam.result_positions[beam.neutralization_index], beam.FI[beam.neutralization_index], 'y--',
             label="Potential at neutralization point")
    plt.plot(beam.result_positions[-1], beam.FI[-1], 'r', label="Final potential")

    q_min = min(beam.FI[0])
    q_max = max(beam.FI[0])
    r_min = 0
    r_max = beam.r
    plt.ylabel("Potenciál [V]")
    plt.xlabel("Radiális pozíció [m]")
    plt.axis([r_min, r_max, q_min, q_max])
    plt.legend()
    plt.show()

beamDescription = BeamDescription.get_description_from_files(
    open("beamdescription/beam_1_profile.dat", "r"),
    open("beamdescription/beam.dat", "r"))


def main():
    basic_beam(2.5, 40)


def basic_beam(I, E, p=0, T=0):
    if p == 0:
        gas_density = 0
    else:
        gas_density = p / (scipy.constants.k * (273.15 + T))
    print(gas_density)
    beamDescription = BeamDescription.get_description_from_files(
        open("beamdescription/beam_1_profile.dat", "r"),
        open("beamdescription/beam.dat", "r"))
    calculator = BeamCalculator(beamDescription, I, E, r=5, e_v_r=0, gas_density=gas_density)
    beam = calculator.calculate_beam(r_resolution=100, z_interval=1)
    plot_beam(beam)
    plt.clf()

main()
