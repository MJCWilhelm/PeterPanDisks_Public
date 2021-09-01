import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as clr
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

from FRIED_interp import FRIED_interpolator


MassJupSun = 9.55e-4


def plot_trajectories (disk_radii, disk_masses, logR, logM, label):
    '''
    Plot the evolution of a series of disks through the disk radius - disk mass fraction plane

    disk_radii: radii of disks, shape (N, M), N number of snapshots, M number of disks
    disk_masses: masses disks, shape as above
    logM: unique disk mass fractions on FRIED grid
    logR: unique disk radii on FRIED grid
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot FRIED data points
    ax.scatter(logR, logM, c='b', s=3)


    # Plot trajectories
    cmap = plt.get_cmap('plasma')
    cmap_norm = clr.Normalize(vmin=0, vmax=disk_radii.shape[1])
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    for i in range(disk_radii.shape[1]):
        ax.plot(np.log10(disk_radii[:,i]), np.log10(disk_masses/np.logspace(np.log10(0.08), np.log10(1.9), num=disk_radii.shape[1]))[:,i], c=scalarmap.to_rgba(i), linewidth=0.3)

    # Plot initial mass/radius line
    host_star_mass = np.logspace(np.log10(0.08), np.log10(1.9), num=disk_radii.shape[1])
    init_disk_radii = 200. * host_star_mass**0.45
    init_disk_mass_frac = 0.24 * host_star_mass**-0.27

    ax.plot(np.log10(init_disk_radii), np.log10(init_disk_mass_frac), c='k')

    ax.set_xlabel('log10 R$_d$ [au]')
    ax.set_ylabel('log10 M$_d$/M$_*$')

    ax.set_xlim(-0.5, 3)
    ax.set_ylim(-8, 0)

    # Fill convex hull of the logarithmic grid
    x = [
        np.min(logR),
        np.min(logR),
        np.max(logR),
        np.max(logR)
    ]

    y = [
        np.min(logM),
        np.max(logM[ logR == np.min(logR) ]),
        np.max(logM),
        np.min(logM[ logR == np.max(logR) ])
    ]

    ax.fill(x, y, c='b', alpha=0.1)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.text(-0.3, -0.8, label)


# Paper figure 5
Mstar, F, Mdisk, _, Rdisk, Mdot = np.loadtxt('friedgrid.dat', unpack=True)

disk_radii_1  = np.loadtxt('../data/F1E1_A1E-3/disk_radius.dat')
disk_masses_1 = np.loadtxt('../data/F1E1_A1E-3/gas_mass.dat')

disk_radii_2  = np.loadtxt('../data/F1E1_A1E-4/disk_radius.dat')
disk_masses_2 = np.loadtxt('../data/F1E1_A1E-4/gas_mass.dat')


Mstar_unique = np.unique(Mstar)

mask = Mstar == Mstar_unique[1]

logR = np.log10(Rdisk[mask])
logM = np.log10(Mdisk[mask]/Mstar_unique[1]*MassJupSun)


plot_trajectories(disk_radii_1, disk_masses_1, logR, logM, '$\\alpha=10^{-3}$')

plot_trajectories(disk_radii_2, disk_masses_2, logR, logM, '$\\alpha=10^{-4}$')



# Paper figure 1
mask = (F == 10.)*(Rdisk == 100.)

# Figure out range of disk masses covered for all stellar masses
Mdisk_min = np.max(Mdisk[ mask*(Mstar == np.min(Mstar)) ])
Mdisk_max = np.min(Mdisk[ mask*(Mstar == np.max(Mstar)) ])

n = 5
Mstar_range = np.logspace(np.log10(0.08), np.log10(np.max(Mstar)), num=100)
Mdisk_range = np.logspace(np.log10(Mdisk_min), np.log10(Mdisk_max), num=n)

cmap_norm = clr.Normalize(vmin=0, vmax=n)
scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=plt.get_cmap('plasma_r'))

interpolator = FRIED_interpolator(verbosity=True)


plt.figure()

plt.plot(Mstar_range, 10.**(1.81*np.log10(Mstar_range) - 8.25), label='Accretion (Alcala 2014)', c='C2')
plt.plot(Mstar_range, 6.25e-9 * Mstar_range**-0.068 * ((10.**(30. + 1.7*np.log10(Mstar_range)))/1e30)**1.14, 
    label='IPE (Owen 2012)', linestyle=':', c='C0')
plt.plot(Mstar_range, 10.**(-2.7326*np.exp((np.log(30. + 1.7*np.log10(Mstar_range)) - 3.3307)**2./(-2.9868*10.**-3)) - 7.2580), 
    label='IPE (Picogna 2019)', linestyle=':', c='#66CCEE')


for i in range(len(Mdisk_range)-1):
    plt.plot(Mstar_range, interpolator.interp(Mstar_range, 10., Mdisk_range[i], 100.), c=scalarmap.to_rgba(i+1), linestyle='--')
plt.plot(Mstar_range, interpolator.interp(Mstar_range, 10., Mdisk_range[-1], 100.), 
    label='EPE (Haworth 2018)', c=scalarmap.to_rgba(len(Mdisk_range)-1), linestyle='--')


plt.xscale('log')
plt.yscale('log')

plt.xlim(0.08, 1.9)
plt.ylim(1e-11, 1e-6)

plt.xlabel('M$_*$ [M$_\\odot$]')
plt.ylabel('$\\dot{\\mathrm{M}}_\\mathrm{d}$ [M$_\\odot$ yr$^{-1}$]')

plt.legend(loc='upper right', frameon=False)


plt.show()
