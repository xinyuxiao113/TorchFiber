"""
Jax Simulation of optical fiber transmission.

group:
- pulse
- SymbTx  
- SignalRx
- Rx(sps=2,chid=0)
    - info            # [Pch, Fi, Rs, Nch]
    - Tx
    - Rx
- Rx(sps=2,chid=0,method=filtering)
    - info            # [Pch, Fi, Rs, Nch]
    - Tx
    - Rx
...
"""

import h5py
import argparse,numpy as np,  os, pickle, yaml, torch, time
from src.TorchSimulation.transmitter import simpleWDMTx,choose_sps
from src.TorchSimulation.channel import manakov_ssf, choose_dz, get_beta2
from src.TorchSimulation.receiver import simpleRx
from src.TorchDSP.baselines import CDC, DDLMS
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--path',   type=str, default='dataset/test.h5', help='dataset path', required=True)
parser.add_argument('--rx_sps',   type=int, default=2, help='Rx sps')
parser.add_argument('--rx_chid',   type=int, default=0, help='receiver Nch id  respect to center channel')
parser.add_argument('--device',   type=str, default='cuda:0', help='device')
parser.add_argument('--method', type=str, default='frequency cut', help='frequency cut or filtering', required=False)
args = parser.parse_args()


t0 = time.time()
with h5py.File(args.path,'a') as f:
    for key in f.keys():
        group = f[key]
        rx_seed = group.attrs['rx_seed']
        trans_data = torch.from_numpy(group['SignalRx'][...])
        Nch = group.attrs['Nch']


        if f'Rx(sps={args.rx_sps},chid={args.rx_chid},method={args.method})' in group.keys():
            print(f'Rx(sps={args.rx_sps},chid={args.rx_chid},method={args.method}) already exists in {key}')
            continue

        config = {'Rs': group.attrs['Rs(GHz)']*1e9, 'freqspace': group.attrs['freqspace(Hz)'], 'pulse': torch.from_numpy(group['pulse'][...]), 'Nch': group.attrs['Nch'], 'sps': group['SignalRx'].attrs['sps']} 
        rx_data = simpleRx(rx_seed, trans_data, config, Nch//2 + args.rx_chid, rx_sps=args.rx_sps, method=args.method, device=args.device)

        subgrp = group.create_group(f'Rx(sps={args.rx_sps},chid={args.rx_chid},method={args.method})')
        subgrp.attrs['seed'] = rx_seed
        subgrp.attrs['sps'] = args.rx_sps
        subgrp.attrs['chid'] = args.rx_chid
        subgrp.attrs['method'] = args.method

        data = subgrp.create_dataset('Tx', data=group['SymbTx'][:,:,Nch//2 + args.rx_chid,:]/np.sqrt(10))
        data.dims[0].label = 'batch'
        data.dims[1].label = 'time'
        data.dims[2].label = 'modes'
        data.attrs.update({'sps': 1, 'start': 0, 'stop': 0, 'Fs(Hz)': group.attrs['Rs(GHz)']*1e9})

        data = subgrp.create_dataset('Rx', data=rx_data['signal'].cpu().numpy())
        data.dims[0].label = 'batch'
        data.dims[1].label = 'time'
        data.dims[2].label = 'modes'
        data.attrs.update({'sps':args.rx_sps, 'start': 0, 'stop': 0, 'Fs(Hz)': group.attrs['Rs(GHz)']*1e9 * args.rx_sps,})

        info = torch.tensor([group.attrs['Pch(dBm)'], group.attrs['Fc(Hz)'] + args.rx_chid*group.attrs['freqspace(Hz)'],  group.attrs['Rs(GHz)']*1e9 * args.rx_sps, Nch//2 + args.rx_chid])
        info = info.repeat(rx_data['signal'].shape[0], 1)
        data = subgrp.create_dataset('info', data=info.cpu().numpy())
        data.dims[0].label = 'batch'
        data.dims[1].label = 'task: Pch, Fi, Rs, Nch'


t1 = time.time()
print('Rx data genetaion finished.')
print(f'Rx time: {t1-t0}', flush=True)