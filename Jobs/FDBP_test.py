"""
Train MetaDSP Model with split windows.
"""

import pickle, torch, numpy as np, time, argparse, os , yaml, jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard.writer import SummaryWriter
from src.TorchDSP.core import TorchInput, TorchSignal, TorchTime
from src.TorchDSP.dsp import DSP, LDBP, downsamp, ADF, TestDBP
from src.TorchDSP.train_dbp import Test, Train
from src.TorchDSP.dataloader import  MyDataset
from src.TorchDSP.loss import BER_well, MSE, SNR, Qsq
from src.TorchSimulation.receiver import  BER
from src.JaxSimulation.dsp import BPS, bps, ddpll, cpr, mimoaf
import src.JaxSimulation.adaptive_filter as af
from src.JaxSimulation.core import MySignal, SigTime


with open('configs/metadsp/metadbp.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

net = LDBP(cfg['model_info']['DBP_info']).to(cfg['device'])
conv = downsamp(taps=64, Nmodes=2, sps=2, init='zeros').to(cfg['device'])
optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 3e-3}, {'params': conv.parameters(), 'lr': 3e-3}])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Concat data loader


for Nch, Rs , Pch in [(21, 40, -1), (21, 80, 2), (21, 160, 4)]:
    Nwindow = 10000
    train_datas = []
    for pch in [Pch]:
        train_datas.append(MyDataset(cfg['train_path'], Nch=[Nch], Rs=[Rs], Pch=[pch], Nmodes=2,
                        window_size=cfg['tbpl'] + net.overlaps + conv.overlaps, strides=cfg['tbpl'], Nwindow=Nwindow, truncate=0,
                        Tx_window=True, pre_transform='Rx'))
    dataset = ConcatDataset(train_datas)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
    print(len(train_loader))

    test_info = {'Pch': Pch, 'Rs': Rs, 'Nch': Nch}

    log_path = f'_outputs/log_tensorboard/FDBP/ntaps33/{Rs}G_{Nch}ch'
    model_path = f'_models/NewDBP/{Rs}G_{Nch}ch'
    Train(net, conv, train_loader, optimizer, scheduler, log_path, model_path, epoch_init=0, epochs=30, test_info=test_info, save_log=True, save_model=True, save_interval=1, device=cfg['device'])

    kernel = net.task_mlp.parameter.reshape(2,2, cfg['model_info']['DBP_info']['ntaps']).data.cpu()
    torch.save(kernel, f'_outputs/kernel/{Rs}G_{Nch}ch.pt')