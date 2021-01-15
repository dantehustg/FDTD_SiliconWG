###################################################
#       FDTD Sillicon Waveguide
# Auther: Qixin Hu
# Email:  hqx11@hust.edu.cn
###################################################
# This program use 2d-FDTD (with PML) method to simulate
# sillicon_waveguide
##################################################

import numpy as np
import matplotlib.pyplot as plt
from .device import Device

# Free Space permittivity, permeability and speed of light
# real values
c0 = 3e8
e0 = 8.8419412828e-12
u0 = 1.2566370614e-06

# Space dimension - the resolution of the space
# This number should change with STEPS.
N_FACTOR = 5
# Courant stability number
S_FACTOR = 1 / (2 ** (0.5))
# PML boundary width
bound_width = int(4 * N_FACTOR)
# Smooth order
grid_order = 6


class FDTDWaveGuide:
    def __init__(self, device: Device):
        # GRID parameters (Space grid step = 0.25 micrometer / N_FACTOR)
        self.da = 0.25e-6 / N_FACTOR
        # # The time grid is determed by da, S_FACTOR and c0
        self.dt = S_FACTOR * self.da / c0
        # Here we choose total time steps = 1000
        self.STEPS = 2000

        # # GRID SPACE
        self.Y_dim = device.get_Y_dim
        self.X_dim = device.get_X_dim

        # wavelength
        self.wvlen = 1.55  # micrometers
        self.N_lamdba = int(self.wvlen * 1e-6 / self.da)

        # Device
        self.epsilon = device.get_epsilon
        self.mu = device.get_mu

        #  Initial Grid
        self.Ez = np.zeros((self.X_dim, self.Y_dim))
        self.Ezx = np.zeros((self.X_dim, self.Y_dim))
        self.Ezy = np.zeros((self.X_dim, self.Y_dim))
        self.Hy = np.zeros((self.X_dim, self.Y_dim))
        self.Hx = np.zeros((self.X_dim, self.Y_dim))

        # Initial electric conductivity fiedl and PML condition.
        self.Sigma_x = np.zeros((self.X_dim, self.Y_dim))
        self.Sigma_y = np.zeros((self.X_dim, self.Y_dim))
        self.init_pml()
        self.Sigma_mux = (self.Sigma_x * self.mu) / self.epsilon
        self.Sigma_muy = (self.Sigma_y * self.mu) / self.epsilon

        # Calculate PML update coefficient.
        # We only perform Ez model
        # H field update coefficient
        self.mHy0 = (self.mu - 0.5 * self.dt * self.Sigma_mux) / \
            (self.mu + 0.5 * self.dt * self.Sigma_mux)
        self.mHy1 = (self.dt/self.da) / (self.mu +
                                         0.5 * self.dt * self.Sigma_mux)
        self.mHx0 = (self.mu - 0.5 * self.dt * self.Sigma_muy) / \
            (self.mu + 0.5 * self.dt * self.Sigma_muy)
        self.mHx1 = (self.dt/self.da) / (self.mu +
                                         0.5 * self.dt * self.Sigma_muy)

        # E field update coeficient
        self.mEzx0 = (self.epsilon - 0.5 * self.dt * self.Sigma_x) / \
            (self.epsilon + 0.5 * self.dt * self.Sigma_x)
        self.mEzx1 = (self.dt/self.da) / \
            (self.epsilon + 0.5 * self.dt * self.Sigma_x)
        self.mEzy0 = (self.epsilon - 0.5 * self.dt * self.Sigma_y) / \
            (self.epsilon + 0.5 * self.dt * self.Sigma_y)
        self.mEzy1 = (self.dt/self.da) / \
            (self.epsilon + 0.5 * self.dt * self.Sigma_y)

        # The Ez_prop here is to store the visualization ndarray.
        self.Ez_prop = []

        # Print
        print("All the preparatory parameters are calculated and the simulation starts.........")

    def run(self):
        for t in range(self.STEPS):
            self.update_H(t)
            self.update_E(t)
            # 每40个time_step保存一帧
            if (t + 1) % 40 == 0:
                self.save_frame()
                print(f'Current simulation progress: {t+1}/{self.STEPS}')

    def update_H(self, t):
        # update H field from E
        self.Hy[:-1, :-1] = self.mHy0[:-1, :-1] * self.Hy[:-1, :-1] + self.mHy1[:-1, :-1] * \
            (
                self.Ezx[1:, :-1] - self.Ezx[:-1, :-1] +
                self.Ezy[1:, :-1] - self.Ezy[:-1, :-1]
        )
        self.Hx[:-1, :-1] = self.mHx0[:-1, :-1] * self.Hx[:-1, :-1] - self.mHx1[:-1, :-1] * \
            (
                self.Ezx[:-1, 1:] - self.Ezx[:-1, :-1] +
                self.Ezy[:-1, 1:] - self.Ezy[:-1, :-1]
        )

    def update_E(self, t):
        # update E field from H
        self.Ezx[1:, 1:] = self.mEzx0[1:, 1:] * self.Ezx[1:, 1:] + self.mEzx1[1:, 1:] * \
            (
                -self.Hx[1:, 1:] + self.Hx[1:, :-1]
        )

        self.Ezy[1:, 1:] = self.mEzy0[1:, 1:] * self.Ezy[1:, 1:] + self.mEzy1[1:, 1:] * \
            (
                self.Hy[1:, 1:] - self.Hy[:-1, 1:]
        )

        # add source
        self.Ezx[bound_width, 14*N_FACTOR+1:18*N_FACTOR] = 0.5 * \
            np.sin(2*np.pi*(c0/(self.da*self.N_lamdba))*t*self.dt)
        self.Ezy[bound_width, 14*N_FACTOR+1:18*N_FACTOR] = 0.5 * \
            np.sin(2*np.pi*(c0/(self.da*self.N_lamdba))*t*self.dt)

        self.Ez = self.Ezx + self.Ezy

    def init_pml(self):
        """ Initial PML boundary condition."""

        # 先算出Sigama_max
        sigmamax = (6 * (grid_order + 1) * e0 * c0) / \
            (2 * bound_width * self.da)
        # 因为我们是2D-FDTD，所以一共有2个Boundary factor: bf_x, bf_y
        bf_x = ((self.epsilon[int(self.X_dim/2), bound_width] / e0) *
                sigmamax) / ((bound_width ** grid_order) * (grid_order + 1))
        bf_y = ((self.epsilon[bound_width, int(self.Y_dim/2)] / e0) *
                sigmamax) / ((bound_width ** grid_order) * (grid_order + 1))

        # 计算 PML层的参数，我这里两边都取一样的宽度
        slice_x = bf_x * \
            (
                (np.arange(bound_width+1)+0.5*np.ones(bound_width+1))**(grid_order+1) -
                (np.arange(bound_width+1)-0.5 *
                 np.concatenate((np.array([0]), np.ones(bound_width))))**(grid_order+1)
            )

        slice_y = bf_y * \
            (
                (np.arange(bound_width+1)+0.5*np.ones(bound_width+1))**(grid_order+1) -
                (np.arange(bound_width+1)-0.5 *
                 np.concatenate((np.array([0]), np.ones(bound_width))))**(grid_order+1)
            )
        slice_y = slice_y.reshape(-1, 1)

        # 将slice添加到Sigma_x and Siama_y中
        self.Sigma_x[:, :bound_width+1] = slice_x[::-1]
        self.Sigma_x[:, -(bound_width+1):] = slice_x

        self.Sigma_y[:bound_width+1, :] = slice_y[::-1]
        self.Sigma_y[-(bound_width+1):, :] = slice_y

    def save_frame(self):
        self.Ez_prop.append(self.Ez)

    @property
    def get_Ezprop(self):
        return self.Ez_prop

    @property
    def get_dt(self):
        return self.dt