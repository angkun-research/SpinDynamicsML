import numpy as np
import matplotlib.pyplot as plt

# CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B60_Nk250_NJ30.csv'
# choose from 0.0 to 0.1 with 0.01 increments
sigmas = np.arange(0.0, 0.11, 0.01)
mean_vars = np.array([1.544e-5, 3.698e-5, 6.501e-5, 1.486e-4, 2.153e-4, 3.405e-4, 4.950e-4, 7.261e-4, 6.726e-4, 8.820e-4, 1.066e-3])
rmse = np.array([2.875e-3, 1.343e-2, 1.638e-2, 2.312e-2, 3.173e-2, 3.701e-2, 5.022e-2, 5.970e-2, 5.603e-2, 6.390e-2, 7.455e-2])


