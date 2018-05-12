import numpy as np


def mean(positions, density_values):
    return np.trapz(positions * density_values, positions)


def std(positions, density_values,  m=None):
    if m is None:
        m = mean(positions, density_values)
    return np.sqrt(np.trapz((positions - m) ** 2 * density_values, positions))


def kurtosis(positions, density_values, m=None, var=None):
    if m is None:
        m = mean(positions, density_values)
    if var is None:
        var = std(positions, density_values, m)
    return (np.trapz((positions - m) ** 4 * density_values, positions)) / var ** 4 - 3


def mirror(positions, density_values):
    return np.append(-np.flipud(positions), positions), np.append(np.flipud(density_values), density_values)

def normalize(positions, density_values):
    return density_values/np.trapz(density_values, positions)
