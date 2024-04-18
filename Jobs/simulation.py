"""
Jax Simulation of optical fiber transmission.
"""
import argparse,jax,jax.random as rd,numpy as np,jax.numpy as jnp, os,pickle, yaml
from jax._src.config import config
from src.JaxSimulation.transmitter import simpleWDMTx,choose_sps
from src.JaxSimulation.channel import manakov_ssf, choose_dz, get_beta2
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--config', type=str, default='configs/simulation.yaml', help='Path to the YAML configuration file')
_args = parser.parse_args()
with open(_args.config, 'r') as file:
    args = yaml.safe_load(file)

# make directory;  set calculation precision.
if not os.path.exists(args['path']):
    os.mkdir(args['path'])
if args['precision'] == 'float64':
    config.update("jax_enable_x64", True)  # float64 precision

# Set Global params
M = args['M']                   # QAM format.
Ltotal = args['Ltotal']         # Total Length. [km]
Lspan = args['Lspan']           # Length per span. [km]
alpha = args['alpha']           # Attenuation parameter. [dB/km]
D = args['D']                   # Dispersion parameter.  [ps/nm/km]
Dpmd = args['Dpmd']             # PMD parameter. [ps/sqrt(km)]
PMD = args['PMD']               # Simulation PMD or not. True or False.
Lcorr = args['Lcorr']           # Fiber Correlation length. [km]
Fc = args['Fc']                 # Central frequency [Hz]
gamma = args['gamma']           # Nonlinear parameter. [/W/km]
amp = args['amp']               # Amplifier type. 'edfa', 'ideal', or None. 
NF = args['NF']                 # EDFA Noise Figure. [dB]
beta2 = get_beta2(D, Fc)        # [s^2/km]
spacing_factor = args['SF'] # freqspace/Rs


os.makedirs(args['path'], exist_ok=True)
                
for Nch in args['Nch']:
    for Rs in args['Rs']:
        Rs = Rs * 1e9   # symbol rate [Hz]
        for Pch_dBm in args['Pch']:
            print(f'\n#######      Nch:{Nch}       Rs:{int(Rs/1e9)}GHz     Pch:{Pch_dBm}dBm        #######', flush=True)
            freqspace = Rs * spacing_factor
            key,k_tx,k_ch = rd.split(rd.PRNGKey(args['seed']), 3)
            sps = choose_sps(Nch, freqspace, Rs)
            hz = choose_dz(freqspace, Lspan, Pch_dBm, Nch, beta2, gamma)
            print(f'#######     Tx sps = {sps},  simulation hz={hz}km       #######', flush=True)
            

            ## Step 1: Tx
            tx_path = args['path'] + f'/Tx_Nch{Nch}_{int(Rs/1e9)}GHz_Pch{Pch_dBm}dBm'
            channel_path = args['path'] + f'/Channel_Nch{Nch}_{int(Rs/1e9)}GHz_Pch{Pch_dBm}dBm'

            symb_only = False
            tx_data = simpleWDMTx(symb_only, k_tx, args['batch'], M, args['Nbits'], sps, Nch, args['Nmodes'], Rs, freqspace, Pch_dBm)
            with open(tx_path, 'wb') as file: pickle.dump(tx_data['config'], file)
            # jax.profiler.save_device_memory_profile(f"profs4/tx_{Nch}_{Rs}_{Pch_dBm}.prof")

            ## Step 2: channel
            trans_data = manakov_ssf(tx_data, k_ch, Ltotal, Lspan, hz, alpha, D, gamma, Fc,amp, NF,order=1, PMD=PMD)
            with open(channel_path, 'wb') as file: pickle.dump(trans_data, file)
            # jax.profiler.save_device_memory_profile(f"profs4/ch_{Nch}_{Rs}_{Pch_dBm}.prof") 