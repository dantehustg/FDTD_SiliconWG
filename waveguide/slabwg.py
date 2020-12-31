###################################################
#       FDTD Slab Sillicon Waveguide
# Auther: Qixin Hu
# Email:  hqx11@hust.edu.cn
###################################################
# This program use FDTD (with PML) method to simulate
# slab_waveguide, the results will show through .png
# and .gif.
##################################################

import numpy as np
import matplotlib.pyplot as plt

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


class SlabWG:
    def __init__(self):
        # GRID parameters (Space grid step = 0.25 micrometer / N_FACTOR)
        self.da = 0.25e-6 / N_FACTOR
        # # The time grid is determed by da, S_FACTOR and c0
        self.dt = S_FACTOR * self.da / c0
        # Here we choose total time steps = 1000
        self.STEPS = 1000

        # # GRID SPACE
        self.Y_dim = 32 * N_FACTOR
        self.X_dim = 80 * N_FACTOR

        # wavelength
        self.wvlen = 2  # micrometers
        self.N_lamdba = int(self.wvlen * 1e-6 / self.da)

        # Index of WaveGuide(Sillicon)
        self.index = 1.5

        # Add device
        self.epsilon = e0 * np.ones((self.X_dim, self.Y_dim))
        self.mu = u0 * np.ones((self.X_dim, self.Y_dim))
        self.epsilon[:, 14*N_FACTOR+1:18 *
                     N_FACTOR] = self.index * self.index * e0

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

    def run(self):
        # fig, ax = plt.subplots()
        self.Ez[bound_width, 14*N_FACTOR] = 1
        for t in range(self.STEPS):
            self.update_H(t)
            self.update_E(t)

            # 每20个time_step保存一帧
            if (t + 1) % 20 == 0:
                self.save_frame()

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


if __name__ == "__main__":
    # Run this sample
    import os
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani

    # 首先是mkdirs
    path = os.path.join(os.getcwd(), 'results', 'slab_waveguide')
    if not os.path.exists(path):
        os.makedirs(path)

    # 然后运行sample
    sample = SlabWG()
    # 这个时候已经创建好文件夹并且生成了不同时间的frame
    sample.run()

    # 接下来要制作动图
    Ez_prop = sample.get_Ezprop
    dt = sample.get_dt

    def update(n):
        image = plt.imshow(
            Ez_prop[n], interpolation='bilinear', vmax=1, vmin=-0.1, cmap='plasma')
        plt.title("FDTD with PML waveguide simulation, {:.1f}fs".format(
            dt * n * 20 * 1e15))
        plt.xlabel("x (0.05 um)")
        plt.ylabel("y (0.05 um)")
        # 将这一帧的照片也保存下来
        plt.savefig(os.path.join(
            'results', 'slab_waveguide', str(n*20)+'.png'))
        return image,

    fig = plt.figure()
    animation = ani.FuncAnimation(
        fig, update, frames=range(len(Ez_prop)), blit=True)
    animation.save(os.path.join(path, 'Slab_WG.gif'),
                   writer=ani.PillowWriter(fps=5))
