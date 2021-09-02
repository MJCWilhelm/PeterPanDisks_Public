import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as clr
import matplotlib.cm as cm

from amuse.units import units, constants


# Peter Pan disk data from Silverberg et al. 2020 (first six, final one from Lee et al. 2020)
Silverberg2020_age = np.array([45., 42., 42., 42., 45., 45., 55.])
Silverberg2020_age_upper = np.array([11., 6., 6., 6., 11., 11., 11.])
Silverberg2020_age_lower = np.array([7., 4., 4., 4., 7., 7., 7.])

logSilverberg2020_accr = np.array([-9.75, -10.80, -10.9, -10.6, -9.3, -9.9, np.log10(1.3)-10.])
logSilverberg2020_accr_upper = np.array([0.25, 0.07, 0.4, 0.4, 0.4, 0.4, 0.4])
logSilverberg2020_accr_lower = np.array([0.25, 0.05, 0.4, 0.4, 0.4, 0.4, 0.4])

Silverberg2020_accr = 10.**logSilverberg2020_accr
Silverberg2020_accr_upper = 10.**(logSilverberg2020_accr_upper + logSilverberg2020_accr) - 10.**(logSilverberg2020_accr)
Silverberg2020_accr_lower = 10.**(logSilverberg2020_accr) - 10.**(logSilverberg2020_accr - logSilverberg2020_accr_lower)



def load_data (EPE, A, folder='../data/'):
    '''
    load data

    EPE: EPE rates to load (array)
    A: viscosity to load (array)
    folder: where to find data
    '''

    # load multiple files if EPE or A has more than one element
    var_coord = np.argmax([len(EPE), len(A)])
    N = np.max([len(EPE), len(A)])

    coords = np.zeros(2, dtype=int)

    t_array = np.loadtxt(folder+'EPE{a}_A{c}/time.dat'.format(
        a=EPE[0], c=A[0]))

    stellar_masses = np.loadtxt(folder+'EPE{a}_A{c}/disp_times.dat'.format(
        a=EPE[0], c=A[0]), unpack=True)[0]

    disk_mass = np.zeros((N, len(t_array), len(stellar_masses)))
    accretion_rate = np.zeros((N, len(t_array), len(stellar_masses)))
    disk_radius = np.zeros((N, len(t_array), len(stellar_masses)))
    epe_rate = np.zeros((N, len(t_array), len(stellar_masses)))

    disp_times = np.zeros((N, len(stellar_masses)))
    accr_times = np.zeros((N, len(stellar_masses)))

    for i in range(N):

        coords[var_coord] = i

        disk_mass[i] = np.loadtxt(folder+'EPE{a}_A{c}/gas_mass.dat'.format(
            a=EPE[coords[0]], c=A[coords[1]]))
        accretion_rate[i] = np.loadtxt(
            folder+'EPE{a}_A{c}/accr_rate.dat'.format(
            a=EPE[coords[0]], c=A[coords[1]]))
        disk_radius[i] = np.loadtxt(
            folder+'EPE{a}_A{c}/disk_radius.dat'.format(
            a=EPE[coords[0]], c=A[coords[1]]))
        epe_rate[i] = np.loadtxt(
            folder+'EPE{a}_A{c}/epe_rate.dat'.format(
            a=EPE[coords[0]], c=A[coords[1]]))

        disp_times[i] = np.loadtxt(
            folder+'EPE{a}_A{c}/disp_times.dat'.format(a=EPE[coords[0]],
            c=A[coords[1]]),
            unpack=True)[1]

        # find moment at which accretion rates drops below 1e-14 MSun/yr (effectively 0)
        accr_times[i] = t_array[ np.argmax(accretion_rate[i] < 1e-14, axis=0) ]
        mask_nogap = accretion_rate[i,-1] > 1e-14
        accr_times[i, mask_nogap ] = disp_times[i, mask_nogap ]


    disp_times[ disp_times < 0. ] = np.nan
    accr_times[ accr_times < 0. ] = np.nan

    DATA = {
        't': t_array,
        'Mstar': stellar_masses,
        'Mdisk': disk_mass,
        'Mdot_accr': accretion_rate,
        'Rdisk': disk_radius,
        'Mdot_epe': epe_rate,
        't_disp': disp_times,
        't_accr': accr_times,
        'EPE': EPE,
        'A': A,
    }

    return DATA


def plot_stellarmass_lifetime (DATA, t_disp_10G0):
    '''
    plot lifetimes (dispersal and cessation of accretion) as a function of stellar mass
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(DATA['Mstar'], DATA['t_disp'][0], label='$10^{-8}$ M$_\\odot$ yr$^{-1}$', 
        marker='.', linewidth=1, c='C2', markersize=10)
    ax.plot(DATA['Mstar'], DATA['t_disp'][1], label='$10^{-9}$ M$_\\odot$ yr$^{-1}$', 
        marker='.', linewidth=1, c='C1', markersize=10)
    ax.plot(DATA['Mstar'], DATA['t_disp'][2], label='$10^{-10}$ M$_\\odot$ yr$^{-1}$', 
        marker='.', linewidth=1, c='C0', markersize=10)
    ax.plot(DATA['Mstar'], t_disp_10G0, label='10 G$_0$', linestyle=':', marker='.',
        linewidth=1, c='C3', markersize=10)

    ax.plot(DATA['Mstar'], DATA['t_accr'][0], markersize=4, linestyle='--',
        marker='^', linewidth=1, c='C2')
    ax.plot(DATA['Mstar'], DATA['t_accr'][1], markersize=4, linestyle='--',
        marker='^', linewidth=1, c='C1')
    ax.plot(DATA['Mstar'], DATA['t_accr'][2], markersize=4, linestyle='--',
        marker='^', linewidth=1, c='C0')

    ax.plot([1e-1], [-1.], c='k', marker='.', markersize=10, linewidth=1, label='Dispersal')
    ax.plot([1e-1], [-1.], c='k', marker='^', markersize=4, linestyle='--', linewidth=1,
        label='Cessation of accretion')


    ax.axhline(50., c='k')

    ax.set_xlim(0.08, 1.9)
    ax.set_ylim(0., 60.)

    ax.set_xscale('log')

    ax.set_xlabel('M$_*$ [M$_\\odot$]')
    ax.set_ylabel('t$_d$ [Myr]')

    ax.legend(loc='lower left', frameon=False, ncol=2)

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(2))

    ax.text(0.1, 54., '$\\alpha=10^{a}$'.format(a='{'+DATA['A'][0][-2:]+'}'))


def plot_time_accrrate (DATA, index=-1):
    '''
    plot accretion rate through time, for those discs that live to at least 50 Myr
    '''

    # initial conditions, needed for nominal accretion rate
    beta = 1.6
    disk_radii = ( 10.**beta * 100.**-0.5 * 0.17/2e-3 * DATA['Mstar']**0.5 )**(1./(beta - 0.5))
    disk_masses = ( (2e-3)**(1./beta) * 0.17**-2. * 100./10. / DATA['Mstar'] )**(1./(1./beta - 2.))
    Tm = 100. * DATA['Mstar']**(1./4.)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap = plt.get_cmap('plasma')
    cmap_norm = clr.LogNorm(vmin=0.08, vmax=1.9)
    scalarmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)


    for i in range(len(DATA['Mstar'])):
        if np.isnan(DATA['t_disp'][index,i]) or DATA['t_disp'][index,i] >= 50.: # plot all non-dispersed, and post-50 Myr disks
            # actual accretion rate
            ax.plot(DATA['t'][::1000], DATA['Mdot_accr'][index,:,i][::1000], c=scalarmap.to_rgba(DATA['Mstar'][i]))

            # estimate nominal accretion rate evolution
            nu = float(DATA['A'][0]) * constants.kB/constants.u * (Tm[i]|units.K)/disk_radii[i]**0.5 * ((disk_radii[i]|units.AU)**3/(constants.G*(DATA['Mstar'][i]|units.MSun)))**0.5
            t_viscous = ((disk_radii[i]|units.AU)**2 / (3.*nu)).value_in(units.Myr)

            Mdot_nom = 10.**( 1.81*np.log10(DATA['Mstar'][i]) - 8.25 ) \
                    * (1. + DATA['t']/t_viscous)**(-3./2.)

            ax.plot(DATA['t'], Mdot_nom, c=scalarmap.to_rgba(DATA['Mstar'][i]), linestyle=':')

    # data from Silverberg et al. 2020 and Lee et al. 2020. Some values offset for visibilty
    ax.errorbar(Silverberg2020_age+3.*np.array([0., 0., 0.2, 0.1, 0.1, 0.2, 0.]), Silverberg2020_accr, fmt='k.', capsize=5, linewidth=1.,
        xerr=np.array([Silverberg2020_age_lower, Silverberg2020_age_upper]),
        yerr=np.array([Silverberg2020_accr_lower, Silverberg2020_accr_upper]))

    ax.set_yscale('log')

    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('$\\dot{\\mathrm{M}}_{accr}$ [M$_\\odot$ yr$^{-1}$]')

    ax.set_xlim(0., 60.)
    ax.set_ylim(1e-13, 1e-8)

    ax.text(3., 5e-13, '$\\alpha=10^{a}$'.format(a='{'+DATA['A'][0][-2:]+'}'))
    ax.text(3., 2e-13, '$\\dot{M}_{EPE}=10^{a}$ M$_\\odot$ yr$^{c}$'.format(a='{'+DATA['EPE'][index][2:]+'}', M='{\\mathrm{M}}', EPE='{EPE}', c='{-1}'))

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))

    if DATA['A'][0] == '1E-4':
        cax = ax.scatter(-10.*np.ones(len(DATA['Mstar'])), 1e-10*np.arange(len(DATA['Mstar'])), c=DATA['Mstar'], cmap=cmap, norm=clr.LogNorm(vmin=0.08, vmax=1.9))
        cbar = fig.colorbar(cax)

        ticks = [0.08, 0.2, 0.5, 1., 1.9]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.set_label('M$_*$ [M$_\\odot$]')
    

if __name__ == '__main__':

    EPE = ['1E-8', '1E-9', '1E-10']

    # load data
    DATA1 = load_data(EPE, ['1E-3'], label='_XEvol')
    DATA2 = load_data(EPE, ['1E-4'], label='_XEvol')

    t_disp_xray_1 = np.loadtxt('../data/F1E1_A1E-3/disp_times.dat', unpack=True)[1]
    t_disp_xray_2 = np.loadtxt('../data/F1E1_A1E-4/disp_times.dat', unpack=True)[1]


    # Paper figure 2
    plot_stellarmass_lifetime (DATA1, t_disp_xray_1)
    plt.savefig('../figures/Fig2_a1e-3.pdf')

    plot_stellarmass_lifetime (DATA2, t_disp_xray_2)
    plt.savefig('../figures/Fig2_a1e-4.pdf')

    # Paper figure 3
    plot_time_accrrate (DATA1)
    plt.savefig('../figures/Fig3_a1e-3.pdf')

    plot_time_accrrate (DATA2)
    plt.savefig('../figures/Fig3_a1e-4.pdf')
