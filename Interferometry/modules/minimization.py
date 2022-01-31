"""
This module contains functions for fitting simulated data to experiemental data
"""
import numpy as np
from scipy.optimize import minimize


def interferogram_objective_function(simulated_signal, measured_signal):
    """
    Defines the difference between simulated and measured interferograms (by least-squares regression)
    ---
    Args:
    ---
    simulated_signal: 1d array of floats
        simulated interferogram samples
    measured_signal: 1d array of floats
        measured interferogram samples
    ---
    Returns:
    ---
    obj_fun: float
        Value of the objective function to be minimized
    """
    obj_fun = np.sum((simulated_signal - measured_signal)**2)

    return obj_fun


def find_best_mixture_of_interferograms(obj_fun, gen_complex_interferogram, measured_signal):
    # define the bounds for the cutoff frequency
    bounds = [(0,15)]
    res = minimize(lambda coeffs: obj_fun(gen_complex_interferogram, measured_signal, *coeffs), x0=np.array([10]), bounds=bounds,
                   method='SLSQP', tol=1e-10, options={'disp': True})

    return res
