# v0.2: you can just run a 2d fdtd with different grid and time.
# Author: Qixin Hu
# Data: 2020-12-14
# This method comes from https://empossible.net/wp-content/uploads/2020/01/Lecture-Implementation-of-2D-FDTD.pdf

# Basic 2D-FDTD update program Done
# Simple Dirichlet Boundary Condition Done
# Perfectly matched layer Done

"""
TODO:Please debug the boundary condition.
     and add TF/SF source
"""

import numpy as np
import matplotlib.pyplot as plt

# for now we use c0 = 1
c0 = 1
e0 = 1
u0 = 1


class FDTD2dGrid:
    """
    This is a simple 2d FDTD update grid.
    v0.1: you can just run.
    """

    def __init__(self, X, dx, Y, dy, TimeSteps):
        # Grid(space) parameters
        self.X = X
        self.dx = dx
        self.Y = Y
        self.dy = dy
        self.TimeSteps = TimeSteps

        self.dt = e0 * min(dx, dy) / (2 * c0)

        # calculate grid number
        self.Nx = int(X / dx)
        self.Ny = int(Y / dy)

        # we only perform Ez mode
        self.Ez = np.zeros((self.Ny, self.Nx))
        self.Dz = np.zeros((self.Ny, self.Nx))  # This Dz here is for Materials
        self.Hx = np.zeros((self.Ny, self.Nx))
        self.Hy = np.zeros((self.Ny, self.Nx))

        # field's curl (where you implement boundary condition)
        self.CEx = np.zeros((self.Ny, self.Nx))
        self.CEy = np.zeros((self.Ny, self.Nx))
        self.CHz = np.zeros((self.Ny, self.Nx))

        # Initialize integration arrays
        self.ICEx = np.zeros((self.Ny, self.Nx))
        self.ICEy = np.zeros((self.Ny, self.Nx))
        self.IDz = np.zeros((self.Ny, self.Nx))

        # for now the epsi and mu all be one
        self.epsi = np.ones((self.Ny, self.Nx))
        self.mu = np.ones((self.Ny, self.Nx))

        # initial PML condition parameters
        self.Sigma_x = np.zeros((self.Ny, self.Nx))
        self.Sigma_y = np.zeros((self.Ny, self.Nx))
        self.Npx = 20
        self.Npy = 20
        self.init_pml()

        # initial update coefficients.
        self.mHx0 = (1 / self.dt) + self.Sigma_y / (2 * e0)
        self.mHx1 = ((1 / self.dt) - self.Sigma_y / (2 * e0)) / self.mHx0
        self.mHx2 = - (1 / self.mHx0) * (c0 / self.mu)
        self.mHx3 = - (1 / self.mHx0) * (c0 * self.dt / e0) * (self.Sigma_x / self.mu)

        self.mHy0 = (1 / self.dt) + self.Sigma_x / (2 * e0)
        self.mHy1 = ((1 / self.dt) - self.Sigma_x / (2 * e0)) / self.mHy0
        self.mHy2 = - (1 / self.mHy0) * (c0 / self.mu)
        self.mHy3 = - (1 / self.mHy0) * (c0 * self.dt / e0) * (self.Sigma_y / self.mu)

        self.mDz0 = (1 / self.dt) + \
                    (self.Sigma_x + self.Sigma_y) / (2 * e0) + self.Sigma_x * self.Sigma_y * (self.dt / (4 * e0 ** 2))
        self.mDz1 = (1 / self.mDz0) * ((1 / self.dt) - (self.Sigma_x + self.Sigma_y) /
                                       (2 * e0) - self.Sigma_x * self.Sigma_y * self.dt / (4 * e0 ** 2))
        self.mDz2 = c0 / self.mDz0
        self.mDz4 = - (1 / self.mDz0) * (self.dt / e0 ** 2) * self.Sigma_x * self.Sigma_y  # Why 4? not 3??

    def init_pml(self):
        """
        Smoothly generate Sigma_x and Sigma_y (PML parameters)
        """
        # initial Sigma_x
        for x in range(self.Npx):
            sigmax = (1 / (2 * self.dt)) * ((x + 1) / self.Npx) ** 3
            self.Sigma_x[:, self.Npx - x - 1] = sigmax
            self.Sigma_x[:, -(self.Npx - x)] = sigmax

        # initial Sigma_y
        for y in range(self.Npy):
            sigmay = (1 / (2 * self.dt)) * ((y + 1) / self.Npy) ** 3
            self.Sigma_y[self.Npy - y - 1, :] = sigmay
            self.Sigma_y[-(self.Npy - y), :] = sigmay

    def run(self):
        fig, ax = plt.subplots()

        for t in range(self.TimeSteps):

            self.update_H()
            self.update_E(t)

            # plot every 10 iteration
            if (t % 20) == 0:
                ax.cla()
                ax.imshow(self.Ez, cmap='seismic')
                plt.title(str(t) + "/" + str(self.TimeSteps) + "Iterations.")
                plt.pause(0.1)

    def update_curlH(self):
        """
        update curl H with dirichlet boundary.
        """

        self.CHz[0, 0] = (self.Hy[0, 0] - 0) / self.dx - \
                         (self.Hx[0, 0] - 0) / self.dy

        self.CHz[1:, 1:] = (self.Hy[1:, 1:] - self.Hy[1:, :-1]) / self.dx - \
                           (self.Hx[1:, 1:] - self.Hx[:-1, 1:]) / self.dy

        self.CHz[0, 1:] = (self.Hy[0, 1:] - self.Hy[0, :-1]) / self.dx - \
                          (self.Hx[0, 1:] - 0) / self.dy

        self.CHz[1:, 0] = (self.Hy[1:, 0] - 0) / self.dx - \
                          (self.Hx[1:, 0] - self.Hx[:-1, 0]) / self.dy

    def update_curlE(self):
        """
        update curl E with dirichlet boundary condition.
        """
        # update CEx with dirichlet boundary
        self.CEx[:-1, :] = (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dy
        self.CEx[-1, :] = (0 - self.Ez[-1, :]) / self.dy

        # update CEy with dirichlet boundary
        self.CEy[:, :-1] = - (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dx
        self.CEy[:, -1] = - (0 - self.Ez[:, -1]) / self.dx

    def update_H(self):
        """
        Update H field
        """
        # firstly we should update CEx and CEy
        self.update_curlE()

        # Then we update the integrations
        self.ICEx += self.CEx
        self.ICEy += self.CEy

        # Finally we update our H field
        self.Hx = self.mHx1 * self.Hx + self.mHx2 * self.CEx + self.mHx3 * self.ICEx
        self.Hy = self.mHy1 * self.Hy + self.mHy2 * self.CEy + self.mHy3 * self.ICEy

    def update_E(self, i):
        """
        Update E field
        """
        # Firstly we should update CHz
        self.update_curlH()

        # Then we update the integration
        self.IDz += self.Dz

        # Finally we update our D field, apply soft source and update E field
        self.Dz = self.mDz1 * self.Dz + self.mDz2 * self.CHz + self.mDz4 * self.IDz

        self.Dz[80, 80] += 100 * np.exp(-(i - 30) * (i - 30) / 100)

        self.Ez = (1 / self.epsi) * self.Dz

    def test(self):
        # self.init_pml()
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.Sigma_x)
        # plt.subplot(1, 2, 2)
        # plt.imshow(self.Sigma_y)
        # plt.show()

        plt.subplot(3, 4, 1)
        plt.imshow(self.mHx0)
        plt.subplot(3, 4, 2)
        plt.imshow(self.mHx1)
        plt.subplot(3, 4, 3)
        plt.imshow(self.mHx2)
        plt.subplot(3, 4, 4)
        plt.imshow(self.mHx3)

        plt.subplot(3, 4, 5)
        plt.imshow(self.mHy0)
        plt.subplot(3, 4, 6)
        plt.imshow(self.mHy1)
        plt.subplot(3, 4, 7)
        plt.imshow(self.mHy2)
        plt.subplot(3, 4, 8)
        plt.imshow(self.mHy3)

        plt.subplot(3, 4, 9)
        plt.imshow(self.mDz0)
        plt.subplot(3, 4, 10)
        plt.imshow(self.mDz1)
        plt.subplot(3, 4, 11)
        plt.imshow(self.mDz2)
        plt.subplot(3, 4, 12)
        plt.imshow(self.mDz4)

        plt.show()

        pass


if __name__ == '__main__':
    FDTD2d = FDTD2dGrid(400, 1, 200, 1, 1000)
    # FDTD2d.test()
    FDTD2d.run()

    pass
