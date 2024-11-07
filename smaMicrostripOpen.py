import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
from skrf.media.definedAEpTandZ0 import DefinedAEpTandZ0
from skrf.media import DefinedGammaZ0
from skrf.media.mline import MLine
from skrf.frequency import Frequency
from scipy.optimize import minimize
from scipy.constants import c, pi

# Settings, parameters etc.
model = 'ideal'                                                             # either 'ideal' or 'physical'
params_ideal = {
    'sma_z0': 50.36,
    'sma_len': 10.266e-3,
    'sma_epr': 1.9666,
    'L': 0.244e-9,
    'microstrip_z0': 41.06,
    'microstrip_len': 74.24e-3,
    'microstrip_epr': 3.25,
    'microstrip_A': 0.067,
    'microstrip_tanD': 0.01339,
}
params_physical = {
    'sma_z0': 50.36,
    'sma_len': 10.266e-3,
    'sma_epr': 1.9666,
    'L': 0.244e-9,
    'microstrip_w': 4e-3,
    'microstrip_h': 1.6e-3,
    'microstrip_t': 0.5e-3,
    'microstrip_len': 74.24e-3,
    'microstrip_epr': 3.25,
    'microstrip_rho': 1.68e-8,
    'microstrip_tanD': 0.01339,
}

# Returns the actual scikit-rf network as a function of input parameters
def create_network(params, freq: rf.Frequency, model='ideal', output='full', z0=50, tand_override=None, **kwargs) -> rf.Network:
    """
    Returns the scikit-rf network for the SMA-Microstrip-Open model.
    Model can be 'physical' or 'ideal'
    Output can be 'full', 'sma', 'microstrip', 'm' or 'media'
    """
    # Frequency, ports and load
    fmin = freq.f[0]
    port1 = rf.Circuit.Port(freq, name='port1', z0=50)
    open = rf.Circuit.Open(freq, name='load', z0=50)
    
    # SMA connector, inductor and microstrip networks
    sma = DefinedAEpTandZ0(
        frequency=freq,
        z0=params['sma_z0'],
        ep_r=params['sma_epr'],
        z0_port=50
    ).line(params['sma_len'], 'm', name='skrf_sma')
    
    inductor = DefinedGammaZ0(frequency=freq, z0=50, z0_port=50).inductor(params['L'], name='skrf_inductor')
    
    if model == 'ideal':
        media = DefinedAEpTandZ0(
            frequency=freq,
            z0=params['microstrip_z0'],
            # z0=params['microstrip_z0'] * (1. + 0.1j * params['microstrip_epr'] * params['microstrip_tanD']),
            ep_r=params['microstrip_epr'],
            A=params['microstrip_A'],
            f_A=fmin,
            tanD=params['microstrip_tanD'],
            model='frequencyinvariant',
            # f_ep=fmin,
            # f_low=1e6,
            # f_high=10000e6,
            z0_port=50
        )
    else:
        media = MLine(
            frequency=freq,
            w=params['microstrip_w'],
            h=params['microstrip_h'],
            t=params['microstrip_t'],
            ep_r=params['microstrip_epr'],
            tand=params['microstrip_tanD'],
            f_epr_tand=fmin,
            rho=params['microstrip_rho'],
            model='hammerstadjensen',
            disp='none',
            diel='frequencyinvariant',
            z0_port=50,
        )

    microstrip = media.line(params['microstrip_len'], 'm', name='skrf_microstrip')
    # media

    if output == 'full':
        # Connections
        cnx = [
            [(port1, 0), (sma, 0)],
            [(sma, 1), (inductor, 0)],
            [(inductor, 1), (microstrip, 0)],
            [(microstrip, 1), (open, 0)],
        ]

        circuit = rf.Circuit(cnx, **kwargs)
        return circuit.network
    elif output == 'sma':
        return sma
    elif output == 'inductor':
        return inductor
    elif output == 'microstrip':
        return microstrip
    elif output == 'media':
        return media
    else:
        raise Exception('Output type incorrect')
    
def ep_r(ntwk, L):
    beta_phase = -np.unwrap(np.angle(ntwk.s[:, 1, 0]))
    ep_r = np.power(beta_phase * c / (2 * pi * ntwk.frequency.f * L), 2)
    return ep_r

def tand(ntwk):
    # because lossless would be abs(S11)**2 + abs(S21)**2 = 1
    alpha_per_length = np.abs(ntwk.s[:,1,0]) / (1. - np.abs(ntwk.s[:,0,0]))
    alpha_per_length = (20.0 * np.log10(alpha_per_length)) / -8.686
    return alpha_per_length * c / (pi * 1.0 * ntwk.frequency.f)

# For two-port
def plot_epr_tand(networks: list[rf.Network], L, title, file=None):
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_epr, ax_tand = axs[0], axs[1]

    ax_epr.set_ylabel(r'$\epsilon_{r,eff}$')
    # ax_epr.set_ylim((0.9, 1.1))
    ax_tand.set_xlabel(f'Frequency ({networks[0].frequency.unit})')
    ax_tand.set_ylabel('tanD')
    # ax_tand.set_ylim((0, 1.0))

    for network in networks:
        network_epr = ep_r(network, L)
        network_tand = tand(network)
        ax_epr.plot(network.frequency.f_scaled, network_epr, label=network.name)
        ax_tand.plot(network.frequency.f_scaled, network_tand, label=network.name)
    
    if not file is None:
        plt.savefig(f'{file}')
    
def plot_s(ref, list, title, file=None):
    num_ports = ref.number_of_ports
    if num_ports == 2:
        # params = [(0, 0), (1, 0), (0, 1), (1, 1)]
        params = [(0, 0), (1, 0)]
    else:
        params = [(0, 0)]
    
    # fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
    num_plots = len(params)*2
    fig, axs = plt.subplots(2, num_plots, figsize=(4*num_plots, 8), sharex=True)
    
    # residuals
    delta = [ref/l for l in list]
    for j, d in enumerate(delta):
        d.name = f"{list[j].name} residuals"
    
    for i, (m, n) in enumerate(params):        
        # magnitude
        ax = axs[0, 2*i]
        for l in list:
            l.plot_s_db(ax=ax, m=m, n=n)
        ref.plot_s_db(ax=ax, c='k', ls='--', m=m, n=n)
        ax.set_title(f'S{m+1}{n+1} Magnitude')
        ax.get_legend().remove()
        
        # phase
        ax = axs[0, 2*i+1]
        for l in list:
            l.plot_s_deg(ax=ax, m=m, n=n)
        ref.plot_s_deg(ax=ax, c='k', ls='--', m=m, n=n)
        ax.set_title(f'S{m+1}{n+1} Phase')
        if i == len(params) - 1:
            ax.legend(loc='upper right', fontsize = 7)
        else:
            ax.get_legend().remove()
       
        # magnitude residuals
        ax = axs[1, 2*i]
        for d in delta:
            d.plot_s_db(ax=ax, m=m, n=n)
        ax.set_title(f'S{m+1}{n+1} Magnitude Residuals')
        ax.get_legend().remove()
        
        # phase residuals
        ax = axs[1, 2*i+1]
        for d in delta:
            d.plot_s_deg(ax=ax, m=m, n=n)
        ax.set_title(f'S{m+1}{n+1} Phase Residuals')
        ax.get_legend().remove()
        
    # general
    fig.suptitle(title)
    fig.tight_layout()
    if not file is None:
        fig.savefig(f'{file}')

# Main.
def main():
    # Setup parameter values
    if model == 'ideal':
        params = params_ideal
    else:
        params = params_physical

    params50ohm = params.copy(); params50ohm['microstrip_z0'] = 50.0
    params50ohmNoSkinEffect = params50ohm.copy(); params50ohmNoSkinEffect['microstrip_A'] = 0.0

    # Specify frequency
    f = Frequency(10, 1000, 201, 'MHz')

    # Load AWR data
    awr_sma = rf.Network(f'{path_simulated}/awr_sma.s2p', name='awr_sma')
    awr_inductor = rf.Network(f'{path_simulated}/awr_inductor.s2p', name='awr_inductor')
    awr_microstrip = rf.Network(f'{path_simulated}/awr_microstrip.s2p', name='awr_microstrip')
    awr_microstrip50ohm = rf.Network(f'{path_simulated}/awr_microstrip50ohm.s2p', name='awr_microstrip50ohm')
    awr_microstrip50ohmNoSkinEffect = rf.Network(f'{path_simulated}/awr_microstrip50ohmNoSkinEffect.s2p', name='awr_microstrip50ohmNoSkinEffect')
    awr_full = rf.Network(f'{path_simulated}/awr_full.s1p', name='awr_full')
    
    # Load measured data
    measured_full = rf.Network(f'{path_measured}/measured.s1p', name='measured')

    # Create skrf data
    skrf_sma = create_network(params, f, model=model, output='sma', name='skrf_sma')
    skrf_inductor = create_network(params, f, model=model, output='inductor', name='skrf_inductor')
    skrf_microstrip = create_network(params, f, model=model, output='microstrip', name='skrf_microstrip')
    skrf_microstrip50ohm = create_network(params50ohm, f, model=model, output='microstrip', name='skrf_microstrip50ohm')
    skrf_microstrip50ohmNoSkinEffect = create_network(params50ohmNoSkinEffect, f, model=model, output='microstrip', name='skrf_microstrip50ohmNoSkinEffect')
    skrf_full = create_network(params, f, model=model, output='full', name='skrf_full')

    # Combine skrf sma and inductor with awr microstrip
    skrf_awr_full_mixed = skrf_sma ** skrf_inductor ** awr_microstrip ** rf.Circuit.Open(f, name='load', z0=50)
    skrf_awr_full_mixed.name = 'skrf_awr_full_mixed'
    
    # Plot S parameters
    # plot_results(awr_sma, [skrf_sma], 'sma')
    # plot_results(awr_inductor, [skrf_inductor], 'inductor')
    plot_s(awr_microstrip, [skrf_microstrip], 'Microstrip', 'figures/microstrip_s.png')
    plot_s(measured_full, [awr_full, skrf_full], 'Full', 'figures/full_s.png')

    # Plot epr and tanD
    plot_epr_tand([awr_microstrip50ohmNoSkinEffect, skrf_microstrip50ohmNoSkinEffect], params['microstrip_len'], 'Microstrip', 'figures/microstrip50ohmNoSkinEffect_epr_tand.png')

if __name__ == '__main__':
    main()