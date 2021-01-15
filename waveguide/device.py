###################################################
#       FDTD Device
# Auther: Qixin Hu
# Email:  hqx11@hust.edu.cn
###################################################
#   This program is the device.
##################################################

import numpy as np

# Free Space permittivity, permeability and speed of light
c0 = 3e8
e0 = 8.8419412828e-12
u0 = 1.2566370614e-06


class Device:
    """
    Dielectric device.
    One function: Device.set_epsilon(height, width, loc((loc_x, loc_y)), epsilon)
    """

    def __init__(self, Nx, Ny):
        # Device shape
        self.X_dim = Nx
        self.Y_dim = Ny

        # Empty device
        self.epsilon = e0 * np.ones((self.X_dim, self.Y_dim))
        self.mu = u0 * np.ones((self.X_dim, self.Y_dim))

    def set_epsilon(self, x_loc, y_loc, epsilon):
        # Change device epsilon
        self.epsilon[x_loc[0]:x_loc[1], y_loc[0]:y_loc[1]] = epsilon

    @property
    def get_epsilon(self):
        return self.epsilon

    @property
    def get_mu(self):
        return self.mu

    @property
    def get_X_dim(self):
        return self.X_dim

    @property
    def get_Y_dim(self):
        return self.Y_dim
