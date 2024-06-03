"""
Train MetaDSP Model with split windows.
"""

import pickle, torch, numpy as np, time
t0 = time.time()
import argparse, os , yaml, torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
from src.TorchDSP.core import TorchInput, TorchSignal, TorchTime
from src.TorchDSP.dsp import DSP, LDBP, downsamp, ADF, TestDBP
from src.TorchDSP.dataloader import  MyDataset
from src.TorchDSP.loss import BER_well, MSE, SNR, Qsq
from src.TorchSimulation.receiver import  BER
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
from src.JaxSimulation.dsp import BPS, bps, ddpll, cpr, mimoaf
import src.JaxSimulation.adaptive_filter as af
from src.JaxSimulation.core import MySignal, SigTime
t1 = time.time()

print(f"Import Time: {t1-t0:.2f}s")

def avg_phase(x,y):
    return torch.angle(torch.mean(x*torch.conj(y), dim=1, keepdim=True))

def rotation_free_MSE(x,y):
    # x, y: [batch, L, Nmodes]
    theta = avg_phase(x,y)
    return torch.mean(torch.abs(torch.exp(-1j*theta)*x - y)**2)


# @partial(jax.jit, backend='cpu', static_argnums=(2, 3))   
# def DDLMS_jax(Rx, Tx, taps=32, sps=2):
#     signal = MySignal(val=Rx, t=SigTime(0,0,sps), Fs=0)
#     truth = MySignal(val=Tx, t=SigTime(0,0,1), Fs=0)
#     model = mimoaf(taps=taps, train=lambda n: n<2000, mimofn=af.ddlms, learnable=False)
#     z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True)
#     return z


@partial(jax.jit, backend='cpu', static_argnums=(2,3,4,5))   
def DDLMS_jax(Rx, Tx, taps=32, sps=2, lead_symbols=2000, lr=[1/2**6, 1/2**7]):
    signal = MySignal(val=Rx, t=SigTime(0,0,sps), Fs=0)
    truth = MySignal(val=Tx, t=SigTime(0,0,1), Fs=0)
    model = mimoaf(taps=taps, train=lambda n: n<lead_symbols, mimofn=af.ddlms, learnable=False, mimokwargs={'lr_w': lr[0], 'lr_f':lr[1], 'lr_b':0})
    z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True)
    return z

# Test DBP + ADF
def Test(net, device='cuda:0', taps=32, Pch=2, Nch=21, Rs=80, Nmodes=2, test_path='dataset_A800/test.h5', ber_discard=20000, Nsymb=50000, ADF='XPM-ADF'):
    '''
    Test DBP + ADF.
        net: LDBP module.
        device: cuda device.
        taps: ADF filter length.
        power: power of the signal.
        Nch: number of channels.
        Rs: symbol rate.
        test_path: path to test data.
        ber_discard: discard the first ber_discard samples.
        Nsymb: number of symbols to test.

    Return:
        {'MSE': mse, 'BER': ber, 'Qsq': Qsq(ber)}
    '''

    # load data
    test_data = MyDataset(test_path,  Nch=[Nch], Rs=[Rs], Pch=[Pch], Nmodes=Nmodes,
                        window_size=net.overlaps + (taps//2 - 1) + Nsymb, strides=1, Nwindow=1, truncate=0,
                        Tx_window=True, pre_transform='Rx')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

    for Rx, Tx, info in test_loader:
        break
    signal = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
    symb = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
    info = info.to(device)

    # DBP
    with torch.no_grad():
        y = net(signal, info)
    
    # ADF
    sig_in = jax.numpy.array(y.val[0].cpu().numpy())
    symb_in = jax.numpy.array(symb.val[0, y.t.start//y.t.sps:y.t.stop//y.t.sps].cpu().numpy())
    z = DDLMS_jax(sig_in, symb_in, taps=taps)

    # metric
    z1 = torch.tensor(jax.device_get(z.val[ber_discard:]))
    z2 = torch.tensor(jax.device_get(symb_in[z.t.start:z.t.stop][ber_discard:]))
    mse = MSE(z1, z2)
    ber = np.mean(BER(z1, z2)['BER'])

    return {'MSE': mse, 'BER': ber, 'Qsq': Qsq(ber)} 




def Train(net: nn.Module, conv: nn.Module, train_loader, optimizer, scheduler, log_path: str, model_path: str, epoch_init: int, epochs:int, test_info={}, save_log=True, save_model=True, save_interval=1, device='cuda:0', model_info=None):
    '''
    Train DBP + ADF.
        net: LDBP module.
        conv: ADF module.
        train_loader: DataLoader for training.
        optimizer: optimizer.
        scheduler: scheduler.
        log_path: path to save logs.
        model_path: path to save models.   
        epoch_init: initial epoch.
        epochs: number of epochs.
        test_info: test info. {'Pch':2,'Rs': 80,'Nch':21}
        save_log: save logs or not.
        save_model: save model or not.
        save_interval: save model every save_interval epochs.
        device: cuda device.
    '''

    # setting
    loss_fn = rotation_free_MSE  # MSE
    writer = SummaryWriter(log_path)


    for epoch in range(epoch_init, epoch_init + epochs + 1): 
        N = len(train_loader)
        train_loss = 0
        t0 = time.time()
        for i,(Rx, Tx, info) in enumerate(train_loader):
            signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
            signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
            info = info.to(device)

            y = net(signal_input, info)  # [B, L, N]
            y = conv(y)
            truth = signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
            loss = loss_fn(y.val, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(loss.item(), end='\r', flush=True)
            if save_log: writer.add_scalar('BatchLoss/train', loss.item(), epoch*N+i)
        t1 = time.time()
        scheduler.step()

        res = Test(net, device, taps=32, **test_info)

        if save_log:
            writer.add_scalar('Loss/train', train_loss/N, epoch)
            writer.add_scalar('Loss/test', res['MSE'], epoch)
            writer.add_scalar('Metric/Qsq', res['Qsq'], epoch)
            writer.add_scalar('Metric/BER', res['BER'], epoch)
    
        print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
        print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (res['BER'], res['Qsq'], res['MSE']), flush=True)

        if epoch % save_interval == 0 and save_model:
            ckpt = {
                'dbp_info': model_info,
                'dbp_param': net.state_dict(),
                'conv_param': conv.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(ckpt, model_path + f'/{epoch}.pth')
            print('Model saved')
    print('Training Finished')

    return writer