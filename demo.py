###################################################
#   Demo of 2d-FDTD with PML to simulate sillicon
# waveguide.
# Including two examples:
#   1: Slab waveguide.
#   2: Photonic crystal waveguide.
###################################################

from waveguide import device, fdtdwg

# free space permittivity
e0 = 8.8419412828e-12
# decice the simulate size.
N_FACTOR = 5
# sillicon index.
index = 3.4


def visualization(device_epsilon, Ez_prop, dt, name):
    """
    将device_epsilon, Ez_prop保存为图片/动图
    """
    print(f"Now we're going to visualize the simulation {name}......")
    # 首先是device_epsilon
    plt.imshow(device_epsilon, interpolation='bilinear')
    plt.title("Device (Epsilon)")
    plt.xlabel("x (0.05 um)")
    plt.ylabel("y (0.05 um)")
    path = os.path.join('results', name)
    plt.savefig(os.path.join(path, 'Device_epsilon.png'))

    # 然后是利用Ez_prop保存中间过程的图片并且制作动图
    def update(n):
        image = plt.imshow(
            Ez_prop[n], interpolation='bilinear', vmax=1, vmin=-0.1, cmap='plasma')
        plt.title("FDTD with PML waveguide simulation, {:.1f}fs".format(
            dt * n * 40 * 1e15))
        plt.xlabel("x (0.05 um)")
        plt.ylabel("y (0.05 um)")
        # 将这一帧的照片也保存下来
        plt.savefig(os.path.join(
            'results', name, str(n*40)+'.png'))
        return image,

    fig = plt.figure()
    animation = ani.FuncAnimation(
        fig, update, frames=range(len(Ez_prop)), blit=True)
    animation.save(os.path.join(path, name+'.gif'),
                   writer=ani.PillowWriter(fps=5))
    print('Done!')
    return 0


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib.animation as ani

    # 首先是mkdirs, 两个demo都要创建
    path_slab = os.path.join(os.getcwd(), 'results', 'slab_waveguide')
    if not os.path.exists(path_slab):
        os.makedirs(path_slab)

    path_phoc = os.path.join(os.getcwd(), 'results',
                             'photonic_crystal_waveguide')
    if not os.path.exists(path_phoc):
        os.makedirs(path_phoc)

    # 创建模拟的空间
    Ny = 32 * N_FACTOR
    Nx = 80 * N_FACTOR

    # 首先把slab_waveguide给创建，模拟以及保存好
    slab_wg = device.Device(Nx, Ny)
    slab_wg.set_epsilon((0, -1), (14 * N_FACTOR, 18 *
                                  N_FACTOR), index * index * e0)

    slab_fdtd = fdtdwg.FDTDWaveGuide(slab_wg)
    # 接下来要进行模拟并且制作动图
    slab_fdtd.run()
    Ez_prop = slab_fdtd.get_Ezprop
    dt = slab_fdtd.get_dt

    visualization(slab_wg.get_epsilon, Ez_prop, dt, 'slab_waveguide')

    # 再把photonic_wg给创建好
    photonic_wg = device.Device(Nx, Ny)
    for i in range(0, 80, 4):
        for j in range(0, 32, 4):
            photonic_wg.set_epsilon(
                (i * N_FACTOR+1, (i + 1) * N_FACTOR), (j * N_FACTOR + 1, (j + 1) * N_FACTOR), index * index * e0)
    photonic_wg.set_epsilon(
        (0, -1), (14 * N_FACTOR, 18 * N_FACTOR), index * index * e0)

    photonic_fdtd = fdtdwg.FDTDWaveGuide(photonic_wg)
    photonic_fdtd.run()
    Ez_prop = slab_fdtd.get_Ezprop
    dt = slab_fdtd.get_dt

    visualization(photonic_wg.get_epsilon, Ez_prop,
                  dt, 'photonic_crystal_waveguide')
