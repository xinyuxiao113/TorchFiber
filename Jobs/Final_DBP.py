import os, jax, torch, numpy as np, matplotlib.pyplot as plt
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from functools import partial
from src.JaxSimulation.dsp import BPS, bps, ddpll, cpr, mimoaf
import src.JaxSimulation.adaptive_filter as af
from src.JaxSimulation.core import MySignal, SigTime

from src.TorchDSP.dataloader import MyDataset
from torch.utils.data import DataLoader
from src.TorchDSP.baselines import DBP, get_omega, LinOp, DDLMS, CDC
from src.TorchSimulation.receiver import BER
from src.TorchDSP.dsp import TestDBP, LDBP
from src.TorchDSP.train_dbp import Test, Train
from src.TorchDSP.core import TorchSignal, TorchTime
from src.TorchDSP.loss import  MSE, SNR, Qsq


@partial(jax.jit, backend='cpu', static_argnums=(2,3,4,5))   
def DDLMS_jax(Rx, Tx, taps=32, sps=2, lead_symbols=20000, lr=(1/2**6, 1/2**6)):
    signal = MySignal(val=Rx, t=SigTime(0,0,sps), Fs=0)
    truth = MySignal(val=Tx, t=SigTime(0,0,1), Fs=0)
    model = mimoaf(taps=taps, train=lambda n: n<lead_symbols, mimofn=af.ddlms, learnable=False, mimokwargs={'lr_w': lr[0], 'lr_f':lr[1], 'lr_b':0})
    z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True)
    return z


def DSP(Rx, Tx, info, step=25, xi=0, lr=(1/2**6, 1/2**6), taps=32, calc_N=40000):
    '''
    Rx, Tx, info: [Nsymb*sps, Nmodes], [Nsymb, Nmodes], [4]
    '''
    dbp = DBP(Rx.to('cuda:0'), 2000e3, 2000e3//step, Fs=info[2:3].to('cuda:0'), power_dbm=info[0:1].to('cuda:0')+xi)
    y = DDLMS_jax(dbp.to('cpu').numpy(), Tx.numpy(), taps=taps, lr=lr)
    output = torch.tensor(jax.device_get(y.val))
    discard = (output.shape[0] - calc_N)//2
    return np.mean(BER(output[discard:-discard], Tx[y.t.start:y.t.stop][discard:-discard])['Qsq']), (output, Tx[y.t.start:y.t.stop])


def MetaDSP(Rx, Tx, info, lr=(1/2**6, 1/2**6), step=5, ntaps=401, taps=32, calc_N=40000, device='cuda:0', train_Rs='40', train_Nch='1'):
    '''
    Rx, Tx, info: [Nsymb*sps, Nmodes], [Nsymb, Nmodes], [4]
    '''
    dic = torch.load(f'_models/A_MetaDSP_R1/MetaDSP_Nmodes{Rx.shape[-1]}_step{step}_train{train_Rs}G_nch{train_Nch}/19.pth')
    net = LDBP(dic['dbp_info'])
    net.load_state_dict(dic['dbp_param'])
    net = net.to('cuda:0')
    net.set_ntaps(ntaps)
    net.cuda()
    # DBP
    signal = TorchSignal(val=Rx[None,:], t=TorchTime(0,0,2)).to(device)
    symb = TorchSignal(val=Tx[None,:], t=TorchTime(0,0,1)).to(device)
    info = info[None,:].to(device)
    with torch.no_grad():
        y = net(signal, info)
    
    # ADF
    sig_in = jax.numpy.array(y.val[0].cpu().numpy())
    symb_in = jax.numpy.array(symb.val[0, y.t.start//y.t.sps:y.t.stop//y.t.sps].cpu().numpy())
    z = DDLMS_jax(sig_in, symb_in, taps=taps, lr=lr)

    # metric
    ber_discard = (z.val.shape[0] - calc_N)//2
    z1 = torch.tensor(jax.device_get(z.val[ber_discard:-ber_discard]))
    z2 = torch.tensor(jax.device_get(symb_in[z.t.start:z.t.stop][ber_discard:-ber_discard]))
    mse = MSE(z1, z2)
    ber = np.mean(BER(z1, z2)['BER'])
    metric = {'MSE': mse, 'BER': ber, 'Qsq': Qsq(ber)} 

    return metric['Qsq'], (z1,z2)


def FDBPDSP(Rx, Tx, info, lr=(1/2**6, 1/2**6), step=5, train_Rs=40, train_Nch=1, taps=32, calc_N=40000, device='cuda:0'):
    '''
    Rx, Tx, info: [Nsymb*sps, Nmodes], [Nsymb, Nmodes], [4]
    '''
    dic = torch.load(f'_models/A_MetaDSP_R1/FDBP_Nmodes{Rx.shape[-1]}_{step}_{train_Rs}G_{train_Nch}ch_Pch[-2, 0, 2, 4]/19.pth')
    net = LDBP(dic['dbp_info'])
    net.load_state_dict(dic['dbp_param'])
    net = net.to('cuda:0')
    net.cuda()
    # DBP
    signal = TorchSignal(val=Rx[None,:], t=TorchTime(0,0,2)).to(device)
    symb = TorchSignal(val=Tx[None,:], t=TorchTime(0,0,1)).to(device)
    info = info[None,:].to(device)
    with torch.no_grad():
        y = net(signal, info)
    
    # ADF
    sig_in = jax.numpy.array(y.val[0].cpu().numpy())
    symb_in = jax.numpy.array(symb.val[0, y.t.start//y.t.sps:y.t.stop//y.t.sps].cpu().numpy())
    z = DDLMS_jax(sig_in, symb_in, taps=taps, lr=lr)

    # metric
    ber_discard = (z.val.shape[0] - calc_N)//2
    z1 = torch.tensor(jax.device_get(z.val[ber_discard:-ber_discard]))
    z2 = torch.tensor(jax.device_get(symb_in[z.t.start:z.t.stop][ber_discard:-ber_discard]))
    mse = MSE(z1, z2)
    ber = np.mean(BER(z1, z2)['BER'])
    metric = {'MSE': mse, 'BER': ber, 'Qsq': Qsq(ber)} 

    return metric['Qsq'], (z1,z2)


def Q_path(Rx, Tx, Ntest=10000, stride=10000):
    Q = []
    for t in np.arange(0, Rx.shape[-2] - Ntest, stride):
        Q.append(np.mean(BER(Rx[t:t+Ntest], Tx[t:t+Ntest])['Qsq']))
    return Q

def get_data(Nch, Rs, Pch, Nmodes, Nsymb=200000):
    if Nmodes == 1: assert Nsymb <= 99900
    data = MyDataset('dataset_A800/test.h5', Nch=[Nch], Rs=[Rs], Pch=[Pch], Nmodes=Nmodes, truncate=0,
                 window_size=Nsymb, strides=1, Nwindow=1, Tx_window=True, pre_transform='Rx')
    return data.__getitem__(0)

import pandas as pd
def max_table(results):
    df = pd.DataFrame(results)
    best_results = df.loc[df.groupby(['Rs', 'Nch'])['Qsq'].idxmax()]
    pivot_table = best_results.pivot_table(index='Rs', columns='Nch', values='Qsq')
    return pivot_table


import torch, numpy as np, matplotlib.pyplot as plt
from src.TorchDSP.dsp import TestDBP, LDBP
from src.TorchDSP.train_dbp import Test, Train
from src.TorchDSP.dataloader import  MyDataset
import pandas as pd
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--stps', type=int, default=1)
args = parser.parse_args()
stps = args.stps

print('DBP stps = ', stps)
results = []
for rs in [40, 80, 160]:
    for nch in [1, 3, 5, 7, 9, 11]:
        for pch in range(-8, 7):
            print(f'Nmodes=1, Rs={rs}, Nch={nch}, Pch={pch}')
            Rx, Tx, info = get_data(Nch=nch, Rs=rs, Pch=pch, Nmodes=1, Nsymb=80000)
            Q = np.max([DSP(Rx, Tx, info, step=25*stps, xi=xi,calc_N=40000)[0] for xi in np.arange(-10,1)])
            results.append({'Rs': rs, 'Nch': nch, 'Pch': pch, 'Qsq': Q})

torch.save(results, f'_outputs/paper_img/DBP{stps}_Nmodes1')


results = []
for rs in [40, 80, 160]:
    for nch in [1, 3, 5, 21]:
        for pch in range(-3, 8):
            print(f'Nmodes=2, Rs={rs}, Nch={nch}, Pch={pch}')
            Rx, Tx, info = get_data(Nch=nch, Rs=rs, Pch=pch, Nmodes=2, Nsymb=80000)
            Q = np.max([DSP(Rx, Tx, info, step=25*stps, xi=xi,calc_N=40000)[0] for xi in np.arange(-10,1)])
            results.append({'Rs': rs, 'Nch': nch, 'Pch': pch, 'Qsq': Q})

torch.save(results, f'_outputs/paper_img/DBP{stps}_Nmodes2')
