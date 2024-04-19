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
    - Rx_CDC 
    - Rx_DBP%d
    - Rx_CDCDDLMS
    - Rx_DBP%dDDLMS
    - ...
"""

import h5py
import argparse,numpy as np,  os, pickle, yaml, torch, time
from src.TorchSimulation.transmitter import simpleWDMTx,choose_sps
from src.TorchSimulation.channel import manakov_ssf, choose_dz, get_beta2
from src.TorchSimulation.receiver import simpleRx
from src.TorchDSP.baselines import CDC, DDLMS, DBP
import warnings
warnings.filterwarnings('error')

parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--path',   type=str, default='dataset/test.h5', help='dataset path')
parser.add_argument('--comp',   type=str, default='CDC', help='method for compensation. CDC or DBP')
parser.add_argument('--stps',   type=int, default=1, help='steps per span for DBP. not used for CDC')
parser.add_argument('--rx_grp',   type=str, default='Rx(sps=2,chid=0)', help='Rx group name')
args = parser.parse_args()


with h5py.File(args.path,'a') as f:
    for key in f.keys():
        group = f[key]
        if args.rx_grp not in group.keys():
            raise KeyError(f'{args.rx_grp} not exists in {key}, please run python -m torch_Rx.py to get Rx data.')
        
        subgrp = group[args.rx_grp]
        Rx = torch.from_numpy(subgrp['Rx'][...])
        Tx = torch.from_numpy(subgrp['Tx'][...]).to(torch.complex64)
        Fs = torch.tensor([subgrp['Rx'].attrs['Fs(Hz)']]*Rx.shape[0]).to(torch.float32)

        if args.comp == 'CDC':
            t0 = time.time()
            E = CDC(Rx.to('cuda:0'), Fs.to('cuda:0'),  group.attrs['distance(km)']*1e3)  # [B, Nfft, Nmodes]
            t1 = time.time()
            print(f'CDC time: {t1-t0}', flush=True)
            F = DDLMS(E.to('cpu'), Tx.to('cpu'), sps=subgrp['Rx'].attrs['sps'], lead_symbols=2000)
            t2 = time.time()
            print(f'DDLMS time: {t2-t1}', flush=True)
            
            if 'Rx_CDC' in subgrp.keys():
                print(f'{args.rx_grp} already exists in {key}.')
                continue
            data = subgrp.create_dataset('Rx_CDC', data=E.cpu().numpy())
            data.dims[0].label = 'batch'
            data.dims[1].label = 'time'
            data.dims[2].label = 'modes'
            data.attrs.update({'sps':subgrp['Rx'].attrs['sps'], 'start': 0, 'stop': 0})

            data = subgrp.create_dataset('Rx_CDCDDLMS', data=F.val.cpu().numpy())
            data.dims[0].label = 'batch'
            data.dims[1].label = 'time'
            data.dims[2].label = 'modes'
            data.attrs.update({'sps':F.t.sps, 'start': F.t.start, 'stop': F.t.stop})

        elif args.comp == 'DBP':
            t0 = time.time()
            dz = group.attrs['Lspan(km)'] / args.stps
            power_dbm = group.attrs['Pch(dBm)']
            E = DBP(Rx.to('cuda:0'), group.attrs['distance(km)']*1e3, dz, Fs.to('cuda:0'), power_dbm)  # [B, Nfft, Nmodes]
            t1 = time.time()
            print(f'DBP time: {t1-t0}', flush=True)
            F = DDLMS(E.to('cpu'), Tx.to('cpu'), sps=subgrp['Rx'].attrs['sps'], lead_symbols=2000)
            t2 = time.time()
            print(f'DDLMS time: {t2-t1}', flush=True)

            if f'Rx_DBP{args.stps}' in subgrp.keys():
                print(f'{args.rx_grp} already exists in {key}.')
                continue

            data = subgrp.create_dataset(f'Rx_DBP{args.stps}', data=E.cpu().numpy())
            data.dims[0].label = 'batch'
            data.dims[1].label = 'time'
            data.dims[2].label = 'modes'
            data.attrs.update({'sps':subgrp['Rx'].attrs['sps'], 'start': 0, 'stop': 0})

            data = subgrp.create_dataset(f'Rx_DBP{args.stps}DDLMS', data=F.val.cpu().numpy())
            data.dims[0].label = 'batch'
            data.dims[1].label = 'time'
            data.dims[2].label = 'modes'
            data.attrs.update({'sps':F.t.sps, 'start': F.t.start, 'stop': F.t.stop})




