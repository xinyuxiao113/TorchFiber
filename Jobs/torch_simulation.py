"""
Jax Simulation of optical fiber transmission.
"""
import argparse,numpy as np,  os, pickle, yaml, torch
from src.TorchSimulation.transmitter import simpleWDMTx,choose_sps
from src.TorchSimulation.channel import manakov_ssf, choose_dz, get_beta2
from src.TorchSimulation.receiver import simpleRx
from src.TorchDSP.baselines import CDC, DDLMS
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--config', type=str, default='configs/simulation/torch.yaml', help='Path to the YAML configuration file')
parser.add_argument('--seed',   type=int, default=1232, help='random seed for simulation')
_args = parser.parse_args()
with open(_args.config, 'r') as file:
    args = yaml.safe_load(file)
args['seed'] = _args.seed


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

for Nch in args['Nch']:
    for Rs in args['Rs']:
        Rs = Rs * 1e9   # symbol rate [Hz]
        for Pch_dBm in args['Pch']:
            print(f'\n#######      Nch:{Nch}       Rs:{int(Rs/1e9)}GHz     Pch:{Pch_dBm}dBm        #######', flush=True)
            freqspace = Rs * spacing_factor
            sps = choose_sps(Nch, freqspace, Rs)
            hz = choose_dz(freqspace, Lspan, Pch_dBm, Nch, beta2, gamma)
            print(f'#######     Tx sps = {sps},  simulation hz={hz}km       #######', flush=True)
            
            tx_seed = np.random.randint(0, 2**32)
            ch_seed = np.random.randint(0, 2**32)
            rx_seed = np.random.randint(0, 2**32)

            # tx_path = args['path'] + f'/Tx_Nch{Nch}_{int(Rs/1e9)}GHz_Pch{Pch_dBm}dBm'
            # channel_path = args['path'] + f'/Channel_Nch{Nch}_{int(Rs/1e9)}GHz_Pch{Pch_dBm}dBm'

            ## Step 1: Tx
            tx_data = simpleWDMTx(tx_seed, args['batch'], M, args['Nbits'], sps, Nch, args['Nmodes'], Rs, freqspace, Pch_dBm, device=device)
            # with open(tx_path, 'wb') as file: pickle.dump(tx_data['config'], file)

            ## Step 2: channel
            trans_data = manakov_ssf(tx_data, ch_seed, Ltotal, Lspan, hz, alpha, D, gamma, Fc,amp, NF, order=1, openPMD=PMD, device=device)
            # with open(channel_path, 'wb') as file: pickle.dump(trans_data, file)


            ## Step 3: Rx 
            rx_data = simpleRx(rx_seed, trans_data['signal'], tx_data['config'], Nch//2, rx_sps=2, method='frequency cut', device=device)


            Rx = rx_data['signal']    # [B, L*sps, Nmodes]
            Tx = tx_data['SymbTx'][:,:,tx_data['config']['Nch']//2,:].to(torch.complex64)/np.sqrt(10) # [B, L, Nmodes]
            Fs = torch.tensor([2*tx_data['config']['Rs']]*args['batch'])   # [B,]

            E = CDC(Rx.to('cuda:0'), Fs.to('cuda:0'),  2000e3)  # [B, Nfft, Nmodes]
            F = DDLMS(E.to('cpu'), Tx.to('cpu'), sps=2, lead_symbols=2000)

            info = torch.tensor([Pch_dBm, args['Fc'], Rs*2,  Nch])
            info = info.repeat(F.val.shape[0],1)
            data = (F.val, Tx[:,F.t.start:F.t.stop], info)

            attrs = {
                    'Nmodes': args['Nmodes'],
                    'Rs': int(Rs/1e9),
                    'Nch': Nch,
                    'Pch': Pch_dBm,
                    'samplerate(Hz)': Rs*2,
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
                }

            with h5py.File(args['path'], 'a') as hdf:
                file = hdf.create_group(f'Rs{int(Rs/1e9)}_Nch{Nch}_Pch{Pch_dBm}_{args["seed"]}')
                data = file.create_dataset('Rx', data=Rx.cpu().numpy())
                data.dims[0].label = 'batch'
                data.dims[1].label = 'time'
                data.dims[2].label = 'modes'
                data.attrs.update({'sps':2, 'start': 0, 'stop': 0})
                data = file.create_dataset('Tx', data=Tx.cpu().numpy())
                data.dims[0].label = 'batch'
                data.dims[1].label = 'time'
                data.dims[2].label = 'modes'
                data = file.create_dataset('Rx_CDCDSP', data=F.val.cpu().numpy())
                data.dims[0].label = 'batch'
                data.dims[1].label = 'time'
                data.dims[2].label = 'modes'
                data = file.create_dataset('info', data=info.cpu().numpy())
                data.dims[0].label = 'batch'
                data.dims[1].label = 'task: Pch, Fc, Rs, Nch'
                file.attrs.update(attrs)
                file['Rx_CDCDSP'].attrs.update({'sps':F.t.sps, 'start': F.t.start, 'stop': F.t.stop})

            

            

