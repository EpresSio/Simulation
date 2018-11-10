import sys

from BeamCalculator import BeamCalculator
from BeamDescriptor import BeamDescription
import numpy as np

import matplotlib.pyplot as plt


def plot_beam(beam):
    plt.title("Beam shape \nI = " + str(beam.I * 1e3) + " mA, E = " + str(
        beam.energy) + " keV, Gas density = " + str(beam.gas_density) + " m^-2\n"
              + "Start point = " + str(beam.start_point[0]) + " m, End point = " + str(beam.end_point[0]) + " m, "
              + "Neutralization point = " + str(beam.neutralization_point) + " m")

    plt.plot(beam.start_shape()[0], beam.start_shape()[1], "k", label="Initial beam shape")
    plt.plot(beam.before_neutralization_shape()[0], beam.before_neutralization_shape()[1], 'y--',
             label="Beam shape at neutralization point")
    plt.plot(beam.end_shape()[0], beam.end_shape()[1], 'r', label="Final beam shape")

    q_min = min(beam.start_shape()[1])
    q_max = max(beam.start_shape()[1])
    r_min = 0
    r_max = beam.r
    plt.ylabel("Current density [mA/m^3]")
    plt.xlabel("Radial length [m]")
    plt.axis([r_min, r_max, q_min, q_max])
    plt.legend()
    plt.show()


def plot_FI(beam):
    plt.title("Potential \nI = " + str(beam.I * 1e3) + " mA, E = " + str(beam.energy) + " keV\n"
              + "Start point = " + str(beam.start_point[0]) + " m, End point = " + str(beam.end_point[0]) + " m, "
              + "Neutralization point = " + str(beam.neutralization_point) + " m")

    plt.plot(beam.result_positions[0], beam.FI[0], "k", label="Initial potential")
    plt.plot(beam.result_positions[beam.neutralization_index], beam.FI[beam.neutralization_index], 'y--',
             label="Potential at neutralization point")
    plt.plot(beam.result_positions[-1], beam.FI[-1], 'r', label="Final potential")

    q_min = min(beam.FI[0])
    q_max = max(beam.FI[0])
    r_min = 0
    r_max = beam.r
    plt.ylabel("Potential [V]")
    plt.xlabel("Radial length [m]")
    plt.axis([r_min, r_max, q_min, q_max])
    plt.legend()
    plt.show()


# def create_plots_at_current_region(descriptor, I0, I1, interval, E, neutralization_point=50, r=5, r_resolution=30,
#                                    z_interval=1):
#     I_range = np.arange(start=I0, stop=I1 + interval, step=interval)
#
#     FI = []
#     R = 0
#
#     for i in range(len(I_range)):
#         print(str(I_range[i]) + " mA")
#         calculator = BeamCalculator(descriptor, I_range[i], E, r=r, neutralization_point=neutralization_point)
#         beam = calculator.calculate_beam(r_resolution=r_resolution, z_interval=z_interval)
#         if FI == []:
#             R = np.linspace(0, beam.r, r_resolution)
#             FI = np.zeros((len(I_range), r_resolution))
#
#         FI[i] = beam.FI[-1]
#
#     plt.plot(R, FI[-1])
#     plt.show()
#
#     fig = plt.figure(figsize=(20, 10))
#     ax1 = fig.add_subplot(111)
#     ax1.set_title("Proportion of maximums")
#     ax1.set_xlabel("I [mA]")
#     ax1.set_ylabel("r [m]")
#     cs1 = ax1.contourf(I_range, R, np.transpose(FI), 100, cmap=plt.cm.inferno)
#     # cs1.ticks([cs1.vmin, cs1.vmax])
#     cb1 = fig.colorbar(cs1, shrink=0.9)
#     cb1.set_ticks([cb1._boundaries[0], (cb1._boundaries[0]+cb1._boundaries[-1])/2, cb1._boundaries[-1]])
#
#     plt.show()

#
# def create_plots_at_current_region(descriptor, I0, I1, interval, E, neutralization_point=50, r=10, r_resolution=30,
#                                    z_interval=1):
#     I_range = np.arange(start=I0, stop=I1 + interval, step=interval)
#
#     z = []
#     maximum_differences = []
#     std = []
#     kurtosis = []
#
#     for i in range(len(I_range)):
#         print(str(I_range[i]) + " mA")
#         calculator = BeamCalculator(descriptor, I_range[i], E, r=r, neutralization_point=neutralization_point)
#         beam = calculator.calculate_beam(r_resolution=r_resolution, z_interval=z_interval)
#         if len(z) == 0:
#             z = beam.z
#             maximum_differences = np.zeros((len(I_range), len(beam.z)))
#             std = np.zeros((len(I_range), len(beam.z)))
#             kurtosis = np.zeros((len(I_range), len(beam.z)))
#
#         maximum_differences[i] = beam.get_maximum_differences()
#         std[i] = beam.get_std()
#         kurtosis[i] = beam.get_kurtosis()
#
#     fig = plt.figure(figsize=(20, 10))
#
#     ax1 = fig.add_subplot(334)
#     ax1.set_title("Proportion of maximums")
#     ax1.set_xlabel("I [mA]")
#     ax1.set_ylabel("All results\n\n\nz [m]")
#     cs1 = ax1.contourf(I_range, z, np.transpose(maximum_differences), 1000, cmap=plt.cm.inferno)
#     # cs1.ticks([cs1.vmin, cs1.vmax])
#     cb1 = fig.colorbar(cs1, shrink=0.9)
#     cb1.set_ticks([cb1._boundaries[0], (cb1._boundaries[0]+cb1._boundaries[-1])/2, cb1._boundaries[-1]])
#
#
#     ax2 = fig.add_subplot(335)
#     ax2.set_title("Standard deviation")
#     ax2.set_xlabel("I [mA]")
#     ax2.set_ylabel("z [m]")
#     cs2 = ax2.contourf(I_range, z, np.transpose(std), 1000, cmap=plt.cm.inferno)
#     cb2 = fig.colorbar(cs2, shrink=0.9)
#     cb2.set_ticks([cb2._boundaries[0], (cb2._boundaries[0]+cb2._boundaries[-1])/2, cb2._boundaries[-1]])
#
#     ax3 = fig.add_subplot(336)
#     ax3.set_title("Kurtosis")
#     ax3.set_xlabel("I [mA]")
#     ax3.set_ylabel("z [m]")
#     cs3 = ax3.contourf(I_range, z, np.transpose(kurtosis), 1000, cmap=plt.cm.inferno)
#     cb3 = fig.colorbar(cs3, shrink=0.9)
#     high_bound = cb3._boundaries[-1]
#     low_bound = cb3._boundaries[0]
#     cb3.set_ticks([high_bound, (high_bound+low_bound)/2, low_bound])
#     if high_bound > -0.1:
#         cb3.set_ticklabels(["Normal\n" + str(high_bound), (high_bound+low_bound)/2, str(low_bound) + "\nFlat"])
#
#     ax4 = fig.add_subplot(337)
#     ax4.set_xlabel("I [mA]")
#     ax4.set_ylabel("At z =" + str(z[-1]) + " m\n\n\nProportion of maximums")
#     ax4.plot(I_range, np.transpose(maximum_differences)[-1])
#
#     ax5 = fig.add_subplot(338)
#     ax5.set_xlabel("I [mA]")
#     ax5.set_ylabel("Standard deviation")
#     ax5.plot(I_range, np.transpose(std)[-1])
#
#     ax6 = fig.add_subplot(339)
#     ax6.set_xlabel("I [mA]")
#     ax6.set_ylabel("Kurtosis")
#     ax6.plot(I_range, np.transpose(kurtosis)[-1])
#     # a.set_clim(0,1)
#
#     fig.suptitle("Beam statistic \nI: " + str(I0) + " - " + str(I1) + " mA, " + "E = " + str(beam.energy) + " keV, "
#                  + "Start point = " + str(beam.start_point[0]) + " m, End point = " + str(beam.end_point[0]) + " m, "
#                  + "Neutralization point = " + str(beam.neutralization_point) + " m", fontsize= 25)
#     # a.set_ticks([0.5, 0.7])
#     # a.set_ticklabels([0.5,"lol"])
#     fig.subplots_adjust(top = 0.7)
#     plt.tight_layout()
#     plt.show()

#
# def create_plots_at_current_region(descriptor, I0, I1, interval, E, neutralization_point=600, r=10, directory="plots"):
#     I_range = np.arange(start=I0, stop=I1+interval, step=interval)
#     for i in I_range:
#         plt.clf()
#         calculator = BeamCalculator(descriptor, i, E, r=r, neutralization_point=neutralization_point)
#         beam = calculator.calculate_beam(r_resolution=30, z_interval=1)
#         plot_FI(beam)
#         fileName = str(directory) + "/BEAM_I_" + str(i) + ".png"
#         plt.savefig(fileName)
#         plt.clf()
#
#
beamDescription = BeamDescription.get_description_from_files(
    open("beamdescription/beam_1_profile.dat", "r"),
    open("beamdescription/beam.dat", "r"))


def main():
    beamDescription = BeamDescription.get_description_from_files(
        open("beamdescription/beam_1_profile.dat", "r"),
        open("beamdescription/beam.dat", "r"))
    calculator = BeamCalculator(beamDescription, 1, 50, r=5, neutralization_point=50)
    beam = calculator.calculate_beam(r_resolution=100, z_interval=1)
    plot_beam(beam)
    # plot_FI(beam)


# create_plots_at_current_region(beamDescription, 0.5, 1, 0.5, 50)
main()
