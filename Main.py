from Beam import Beam
from BeamDescriptor import BeamDescription
from numpy import trapz

import matplotlib.pyplot as plt


def plot_beam(beam):
    plt.plot(beam.start_charge_density_values, beam.start_positions)
    plt.plot(beam.end_charge_density_values, beam.end_positions, 'r--')
    q_min = min(beam.start_charge_density_values)
    q_max = max(beam.start_charge_density_values)
    r_min = 0
    r_max = beam.r
    plt.axis([q_min, q_max, 0, r_max])
    plt.show()


beamDescription = BeamDescription.get_description_from_files(
    open("beamdescription/beam_1_profile.dat", "r"),
    open("beamdescription/beam.dat", "r"))
beam = Beam(beamDescription, 2e-2, 50, r=10)
beam.calculate_beam(r_resolution=30, z_interval=10)

plot_beam(beam)
