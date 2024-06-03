import h5py
import torch, numpy as np, matplotlib.pyplot as plt
from src.TorchSimulation.receiver import BER
from src.TorchDSP.loss import Qsq
from src.TorchSimulation.utils import show_symb
from src.TorchDSP.dataloader import MyDataset
from torch.utils.data import DataLoader
from src.TorchDSP.dsp import ADF
from src.TorchDSP.core import TorchSignal, TorchTime

def Q_path(Rx, Tx, Ntest=10000, stride=10000):
    Q = []
    for t in  np.arange(0, Rx.shape[-2] - Ntest, stride):
        Q.append(np.mean(BER(torch.tensor(Rx[t:t+Ntest]), torch.tensor(Tx[t:t+Ntest]))['Qsq']))
    return Q

from src.JaxSimulation.dsp import BPS, bps, ddpll, cpr, mimoaf, MetaMIMO
import src.JaxSimulation.adaptive_filter as af, jax
from src.JaxSimulation.core import MySignal, SigTime
from src.JaxSimulation.MetaOptimizer import *

train_data = MyDataset('dataset_A800/train.h5', Nch=[21], Rs=[80], Pch=[5],Nmodes=2,
                        window_size=400, strides=400-15, Nwindow=200, truncate=0,
                        Tx_window=True, pre_transform='Rx_DBP16')
train_loader = DataLoader(train_data, batch_size=20, shuffle=True)

test_data = MyDataset('dataset_A800/test.h5', Nch=[21], Rs=[80], Pch=[5],Nmodes=2,
                        window_size=400, strides=400-15, Nwindow=100, truncate=0,
                        Tx_window=True, pre_transform='Rx_DBP16')
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
for Rx, Tx, info in test_loader:
    print(Rx.shape, Tx.shape, info.shape)
    break

const = np.unique(Tx)

signal = TorchSignal(val=Rx, t=TorchTime(0,0,2))
truth = TorchSignal(val=Tx, t=TorchTime(0,0,1))