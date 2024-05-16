"""
Train MetaDSP Model with split windows.
"""

import pickle, torch, numpy as np, time
t0 = time.time()
import argparse, os , yaml
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


# Test DBP + static filter
def test_model(net, conv, test_loader, device):
    net.eval()
    conv.eval()
    with torch.no_grad():
        N = len(test_loader)
        test_loss = 0
        ber = 0
        for Rx, Tx, info in test_loader:
            signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
            signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
            info = info.to(device)
            y = net(signal_input, info)
            y = conv(y)
            truth = signal_output.val[:, y.t.start:y.t.stop]
            loss = MSE(y.val, truth)

            test_loss += loss.item()
            ber += np.mean(BER(y.val, truth)['BER'])
    return {'MSE': test_loss/N, 'BER': ber/N, 'Qsq': Qsq(ber/N)} 

@partial(jax.jit, backend='cpu', static_argnums=(2))   
def DDLMS_jax(Rx, Tx, taps=32):
    signal = MySignal(val=Rx, t=SigTime(0,0,2), Fs=0)
    truth = MySignal(val=Tx, t=SigTime(0,0,1), Fs=0)
    model = mimoaf(taps=taps, train=lambda n: n<2000, mimofn=af.ddlms, learnable=False)
    z, state = model.init_with_output(jax.random.PRNGKey(0), signal, truth, True)
    return z

# Test DBP + ADF
def Test(net, device='cuda:0', taps=32, power=2, Nch=21, Rs=80, test_path='dataset_A800/test.h5', ber_discard=20000, Nsymb=200000):
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
    test_data = MyDataset(test_path,  Nch=[Nch], Rs=[Rs], Pch=[power], Nmodes=2,
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




def Train(net, conv, train_loader, optimizer, scheduler, log_path, epochs, save_log=True, save_model=True, save_interval=10, device='cuda:0'):
    '''
    Train DBP + ADF.
        net: LDBP module.
        conv: ADF module.
        train_loader: DataLoader for training.
        optimizer: optimizer.
        scheduler: scheduler.
        log_path: path to save logs.
        epochs: number of epochs.
        save_log: save logs or not.
        save_model: save model or not.
        save_interval: save model every save_interval epochs.
        device: cuda device.
    '''

    # setting
    loss_fn = rotation_free_MSE  # MSE
    writer = SummaryWriter(log_path)


    for epoch in range(epochs + 1): 
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

        res = Test(net, device, taps=32, power=2)

        if save_log:
            writer.add_scalar('Loss/train', train_loss/N, epoch)
            writer.add_scalar('Loss/test', res['MSE'], epoch)
            writer.add_scalar('Metric/Qsq', res['Qsq'], epoch)
            writer.add_scalar('Metric/BER', res['BER'], epoch)
    
        print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
        print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (res['BER'], res['Qsq'], res['MSE']), flush=True)

        if epoch % save_interval == 0 and save_model:
            torch.save(net.state_dict(), f'_models/metadsp/train_dbp/net_{epoch}.pth')
            torch.save(conv.state_dict(), f'_models/metadsp/train_dbp/conv_{epoch}.pth')
            print('Model saved')
    print('Training Finished')


with open('configs/metadsp/dbp_info.yaml') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


TBPL = cfg['tbpl']

Qp = {}

for train_pch in [[2]]:
    for d_share in [True]:
        for n_share in [True, False]:
            for D_lr in [0]:
                for N_lr in [1e-2]:
                    for step, dtaps, ntaps in [(5, 5001, 1), (10, 2001, 1), (25, 1001, 1), (50, 501, 1), (100, 251, 1), (200, 125, 1), (400, 65, 1), (800, 33, 1)]:
                    # for step, dtaps, ntaps in [(25, 1001, 101)]:

                        log_path = f'_outputs/log_tensorboard/train_dbp/trainpch{train_pch}_Dshare{d_share}_D{D_lr}_Nshare{n_share}_N{N_lr}_step{step}_dtaps{dtaps}_ntaps{ntaps}'
                        print(log_path)

                        ################################
                        shuffle = True      # shuffle data or not
                        conv_taps = 64      # static conv taps.
                        conv_lr = 3e-2      # static conv learning rate
                        cfg['tbpl'] = 5000
                        cfg['batch_size'] = 2 
                        Nwindow = 10

                        ################################
                        
                        print("tbpl = ", cfg['tbpl'])
                        cfg['DBP_info']['step'] = step
                        cfg['DBP_info']['dtaps'] = dtaps
                        cfg['DBP_info']['ntaps'] = ntaps
                        cfg['DBP_info']['d_share'] = d_share
                        cfg['DBP_info']['n_share'] = n_share

                        net = TestDBP(**cfg['DBP_info'])
                        conv = downsamp(taps=conv_taps, Nmodes=cfg['DBP_info']['Nmodes'], sps=2, init='zeros')
                        net.to(cfg['device'])
                        conv.to(cfg['device'])

                        optimizer = torch.optim.Adam([{'params': net.Dkernel, 'lr': D_lr}, {'params': net.Nkernel, 'lr': N_lr}, {'params': conv.parameters(), 'lr': conv_lr}])
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_step'], gamma=0.5)

                        # data loader
                        print(f'Loading data..., train on {train_pch}') 
                        train_data = MyDataset(cfg['train_path'], Nch=[21], Rs=[80], Pch=train_pch, Nmodes=cfg['DBP_info']['Nmodes'],
                                            window_size=cfg['tbpl'] + net.overlaps + conv.overlaps, strides=cfg['tbpl'], Nwindow=Nwindow, truncate=0,
                                            Tx_window=True, pre_transform='Rx')
                        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=shuffle, drop_last=True)

                        print('data batchs:', len(train_loader))
                        
                        Train(net, conv, train_loader, optimizer, scheduler, log_path, cfg['epochs'], save_log=True, save_model=False, save_interval=10, device=cfg['device'])
                        Qp = [Test(net, 'cuda:0', taps=32, power=p)['Qsq'] for p in range(-3, 8)]

                        # save Qp in tensorboard
                        writer = SummaryWriter(log_path)
                        for i, q in enumerate(Qp):
                            writer.add_scalar('Qp(-3 : 8 dBm)', q, i)


