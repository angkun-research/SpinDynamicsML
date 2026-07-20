import numpy as np
import matplotlib.pyplot as plt

# CSV_PATH = './data/triangular_anisotropy_uniform_Js4_B60_Nk250_NJ30.csv'
# choose from 0.0 to 0.1 with 0.01 increments
sigmas = np.arange(0.0, 0.11, 0.01)
mean_vars = np.array([1.544e-5, 3.698e-5, 6.501e-5, 1.486e-4, 2.153e-4, 3.405e-4, 4.950e-4, 7.261e-4, 6.726e-4, 8.820e-4, 1.066e-3])
rmse = np.array([5.777e-3, 2.835e-2, 3.756e-2, 5.349e-2, 7.299e-2, 8.736e-2, 0.1113, 0.1276, 0.1397, 0.1559, 0.1656])