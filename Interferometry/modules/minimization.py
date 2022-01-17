"""
This module contains functions for fitting simulated data to experiemental data
"""
import numpy as np
from scipy.optimize import minimize


def interferogram_objective_function(gen_complex_interferogram, measured_signal, field_ac_weight, interferometric_ac_weight):
    """
    Defines the difference between simulated and measured interferograms (by least-squares regression)
    """
    simulated_signal = gen_complex_interferogram(field_ac_weight=field_ac_weight, interferometric_ac_weight=interferometric_ac_weight,
                                                 temp_shift=0, plotting=False)
    obj_fun = np.sum((simulated_signal - measured_signal)**2)

    return obj_fun

def find_best_mixture_of_interferograms(obj_fun, gen_complex_interferogram, measured_signal):
    # define the bounds for the cutoff frequency
    bounds = [(0,15)]
    res = minimize(lambda coeffs: obj_fun(gen_complex_interferogram, measured_signal, *coeffs), x0=np.array([10]), bounds=bounds,
                   method='SLSQP', tol=1e-10, options={'disp': True})
    #plt.plot(170e12*fc, self.objective_function(170e12*fc, filter_order=4), "+")
    #plt.show()

    return res
