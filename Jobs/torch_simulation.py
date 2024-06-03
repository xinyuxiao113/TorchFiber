"""
Torch Simulation of optical fiber transmission.
"""
import argparse,numpy as np,  os, pickle, yaml, torch, time
from src.TorchSimulation.transmitter import simpleWDMTx,choose_sps
from src.TorchSimulation.channel import manakov_ssf, choose_dz, get_beta2
from src.TorchSimulation.receiver import simpleRx
from src.TorchDSP.baselines import CDC, DDLMS
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--path',   type=str, default='dataset/train.h5', help='dataset path', required=True)
parser.add_argument('--config', type=str, default='configs/simulation/torch_train.yaml', help='Path to the YAML configuration file', required=True)
_args = parser.parse_args()
with open(_args.config, 'r') as file:
    args = yaml.safe_load(file)

args['seed'] = int((time.time() * 1e6) % 1e9)  # 转换时间戳为整数种子


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
device  =  'cuda:0'
# os.makedirs(args['path'], exist_ok=True)
np.random.seed(args['seed'])

import h5py

t0 = time.time()

for Nch in args['Nch']:
    for Rs in args['Rs']:
        Rs = Rs * 1e9   # symbol rate [Hz]
        for Pch_dBm in args['Pch']:
            print(f'\n#######      Nch:{Nch}       Rs:{int(Rs/1e9)}GHz     Pch:{Pch_dBm}dBm        #######', flush=True)
            freqspace = Rs * spacing_factor

            sps = choose_sps(Nch, freqspace, Rs)
            if 'tx_sps_scale' in args.keys():
                sps = Nch * args.get('tx_sps_scale', 2)
                
            hz = choose_dz(freqspace, Lspan, Pch_dBm, Nch, beta2, gamma)
            print(f'#######     Tx sps = {sps},  simulation hz={hz}km       #######', flush=True)
            
            tx_seed = np.random.randint(0, 2**32)
            ch_seed = np.random.randint(0, 2**32)
            rx_seed = np.random.randint(0, 2**32)

            ## Step 1: Tx
            tx_data = simpleWDMTx(tx_seed, args['batch'], M, args['Nbits'], sps, Nch, args['Nmodes'], Rs, freqspace, Pch_dBm, device=device)

            ## Step 2: channel
            trans_data = manakov_ssf(tx_data, ch_seed, Ltotal, Lspan, hz, alpha, D, gamma, Fc,amp, NF, order=1, openPMD=PMD, device=device)


            attrs = {
                    'Nmodes': args['Nmodes'],
                    'Nch': Nch,
                    'Rs(GHz)': int(Rs/1e9),
                    'Pch(dBm)': Pch_dBm,
                    'Lspan(km)': Lspan,
                    'Fc(Hz)': args['Fc'],
                    'distance(km)': Ltotal,
                    'beta2(s^2/km)': beta2,
                    'gamma(/W/km)': gamma,
                    'alpha(dB/km)': alpha,
                    'D(ps/nm/km)': D,
                    'Dpmd(ps/sqrt(km))': Dpmd,
                    'NF(dB)': NF,
                    'amp': amp,
                    'PMD': PMD,
                    'Lcorr(km)': Lcorr,
                    'M(QAM-order)': M, 
                    'batch': args['batch'],
                    'tx_sps': sps,
                    'freqspace(Hz)': freqspace,
                    'rx_seed': rx_seed,
                }
            {'Rs', 'freqspace', 'pulse', 'Nch', 'sps'}
            

            with h5py.File(_args.path, 'a') as hdf:
                if f'Nmodes{args["Nmodes"]}_Rs{int(Rs/1e9)}_Nch{Nch}_Pch{Pch_dBm}_{args["seed"]}' in hdf.keys():
                    print(f'Nmodes{args["Nmodes"]}_Rs{int(Rs/1e9)}_Nch{Nch}_Pch{Pch_dBm}_{args["seed"]} already exists.')
                    continue

                print(f'Creating Dataset: Nmodes{args["Nmodes"]}_Rs{int(Rs/1e9)}_Nch{Nch}_Pch{Pch_dBm}_{args["seed"]}')
                file = hdf.create_group(f'Nmodes{args["Nmodes"]}_Rs{int(Rs/1e9)}_Nch{Nch}_Pch{Pch_dBm}_{args["seed"]}')
                file.attrs.update(attrs)

                if 'save_SignalTx' in args.keys() and args['save_SignalTx'] == True:
                    data = file.create_dataset('SignalTx', data=tx_data['signal'].cpu().numpy())
                    data.dims[0].label = 'batch'
                    data.dims[1].label = 'time'
                    data.dims[2].label = 'modes'
                    data.attrs['sps'] = sps


                data = file.create_dataset('SymbTx', data=tx_data['SymbTx'].cpu().numpy())  
                data.dims[0].label = 'batch'
                data.dims[1].label = 'time'
                data.dims[2].label = 'Nch'
                data.dims[3].label = 'modes'

                data = file.create_dataset('pulse', data=tx_data['config']['pulse'].cpu().numpy())
                data.attrs['sps'] = tx_data['config']['sps']

                data = file.create_dataset('SignalRx', data=trans_data['signal'].cpu().numpy())
                data.dims[0].label = 'batch'
                data.dims[1].label = 'time'
                data.dims[2].label = 'modes'
                data.attrs['sps'] = tx_data['config']['sps']

t1 = time.time()
print('simulation is complete!')   
print(f'Total cost time: {t1-t0}.') 

            

