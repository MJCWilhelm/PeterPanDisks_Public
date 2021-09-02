import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm as cm

import time
from os.path import isdir
import sys
import argparse

import FRIED_interp
from disk_class import Disk, setup_disks_and_codes, run_disks, stop_codes
from amuse.units import units, constants


# Habing flux
G0 = 1.6e-3 * units.erg / units.s / units.cm**2


# alpha (viscosity) and EPE rate (mdot) from command line
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', default=1e-3, type=float)
parser.add_argument('--mdot', default=-1., type=float)

args = parser.parse_args()

alpha = args.alpha
outer_photoevap_rate = args.mdot | units.MSun/units.yr

use_radfield = args.mdot < 0. # if mdot is <0, use radiation field of 10 G0


number_of_vaders = 4    # number of parallel workers
number_of_cells = 300   # grid cells of every disk
number_of_disks = 30    # disks per series

# space stellar masses logarithmically, between hydrogen burning limit (0.08 MSun) and FRIED max mass (1.9 MSun)
stellar_masses = np.logspace(np.log10(0.08), np.log10(1.9), num=number_of_disks) | units.MSun

# Uses maximum stable disk masses from Haworth et al. 2020, and disk radii from
# Andrews et al. 2010, as noted in Emsenhuber et al. 2020. 
beta = 1.6
disk_radii = ( 10.**beta * 100.**-0.5 * 0.17/2e-3 * stellar_masses.value_in(units.MSun)**0.5 )**(1./(beta - 0.5)) | units.AU
disk_masses = ( (2e-3)**(1./beta) * 0.17**-2. * 100./10. / stellar_masses.value_in(units.MSun) )**(1./(1./beta - 2.)) | units.MSun


# grid dimensions
r_min = 0.01  | units.AU
r_max = 3000. | units.AU


# directory to write to
folder = '../data/'
if use_radfield:
    folder += 'F1E1'
else:
    folder += 'EPE1E-' + str(int(round(-np.log10(outer_photoevap_rate.value_in(units.MSun/units.yr)))))
folder += '_A1E-' + str(int(round(-np.log10(alpha)))) + '/'

if not isdir(folder):
    print ("Write folder doesn't exist!")
    sys.exit()


# time parameters
t     = 0.  | units.Myr
t_end = 60. | units.kyr
dt    = 1.  | units.kyr # EPE rate update and data write interval


# initialize disks, and set up VADER codes
codes, disks = setup_disks_and_codes(disk_radii, disk_masses, stellar_masses,
    number_of_vaders, number_of_cells, r_min, r_max, alpha, XRayEvol=True)


# set reference EPE rate
for disk in disks:
    disk.outer_photoevap_rate = outer_photoevap_rate


# initialize FRIED interpolator
interpolator = FRIED_interp.FRIED_interpolator()


N_steps = int(t_end/dt)

# data buffer
accretion_rate = np.zeros((N_steps, number_of_disks))
disk_mass = np.zeros((N_steps, number_of_disks))
epe_rate = np.zeros((N_steps, number_of_disks))
disk_radius = np.zeros((N_steps, number_of_disks))

dispersion_time = np.zeros(number_of_disks)-1.


start_main = time.time()

for i in range(N_steps):

    # evolve only active (i.e., non-dipsersed, hasn't failed convergence)
    active_disks = []

    for j in range(number_of_disks):
        if disks[j].disk_active:
            # set EPE rate
            if use_radfield:
                disks[j].outer_photoevap_rate = interpolator.interp_amuse(
                    disks[j].central_mass, 10. | G0, disks[j].disk_gas_mass,
                    disks[j].disk_radius)
            else:
                disks[j].outer_photoevap_rate = outer_photoevap_rate * (disks[j].disk_radius/disk_radii[j])

            active_disks.append(disks[j])

    accreted_mass_init = [ disk.accreted_mass.value_in(units.MSun) for disk in disks ] | units.MSun


    if (i+1)%10 == 0:
        start = time.time()

    # run VADER codes
    run_disks(codes, active_disks, dt)

    t += dt

    # print runtime
    if (i+1)%10 == 0:
        end = time.time()
        print ("Running {c} disks for a step took {a} s, time is {b} Myr".format(
            a=end-start, b=round(t.value_in(units.Myr), 5), c=len(active_disks)), flush=True)


    # actual average accretion rate in timestep
    accreted_mass_after = [ disk.accreted_mass.value_in(units.MSun) for disk in disks ] | units.MSun
    accretion_rate[i] = ((accreted_mass_after - accreted_mass_init)/dt).value_in(units.MSun/units.yr)

    disk_mass[i] = [ disk.disk_gas_mass.value_in(units.MSun) for disk in disks ]

    epe_rate[i] = [ disk.outer_photoevap_rate.value_in(units.MSun/units.yr) for disk in disks ]

    disk_radius[i] = [ disk.disk_radius.value_in(units.AU) for disk in disks ]

    # register time of dispersion per disk
    for j in range(len(disks)):
        if disks[j].disk_active == False and dispersion_time[j] < 0.:
            dispersion_time[j] = t.value_in(units.Myr)


end_main = time.time()

print ("Running {a} disks for {b} Myr took {c} s, {d} min".format(
    a=number_of_disks, b=round(t_end.value_in(units.Myr), 5), 
    c=end_main-start_main, d=(end_main-start_main)/60.))


# timestamps of data
t_array = (np.arange(N_steps)+1)*dt.value_in(units.Myr)


# write data out
np.savetxt(folder+'gas_mass.dat', disk_mass, header='disk gas mass in MSun')
np.savetxt(folder+'accr_rate.dat', accretion_rate, header='disk accretion rate in MSun/yr')
np.savetxt(folder+'epe_rate.dat', epe_rate, header='external photoevaporation in MSun/yr')
np.savetxt(folder+'disk_radius.dat', disk_radius, header='disk radius in AU')
np.savetxt(folder+'disp_times.dat', np.stack([stellar_masses.value_in(units.MSun), dispersion_time]).T,
    header='host star mass in MSun; time of disk dispersion in Myr')
np.savetxt(folder+'time.dat', t_array, header='time series in Myr')
