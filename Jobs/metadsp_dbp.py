"""
Train MetaDSP Model with split windows.
"""
import pickle, torch, numpy as np, time
import argparse, os , yaml
from torch.utils.data import DataLoader
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim import Adam

from src.TorchDSP.core import TorchInput, TorchSignal, TorchTime
from src.TorchDSP.dsp import DSP, LDBP, downsamp, ADF
from src.TorchDSP.layers import ComplexConv1d
from src.TorchDSP.dataloader import signal_dataset, get_data, MyDataset
from src.TorchDSP.loss import BER_well, MSE, SNR, Qsq
from src.TorchSimulation.receiver import  BER

with open('configs/metadsp/fdbp.yaml', 'r') as f:
    config = yaml.safe_load(f)


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


config['device'] = 'cuda:0'
config['tbpl'] = 500
config['batch_size'] = 64


net = LDBP(config['model_info']['DBP_info'])
conv = downsamp(taps=32, Nmodes=2, sps=2)
net.to(config['device'])
conv.to(config['device'])
optimizer = Adam([{'params': net.parameters(), 'lr': 1e-3}, {'params': conv.parameters(), 'lr': 1e-2}])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

window_size = config['tbpl'] + net.overlaps +conv.overlaps
device = config['device']
batch_size = config['batch_size']
epochs = config['epochs']
tbpl = config['tbpl']                       # truncated backpropagation length
lr = config['lr']


train_data = MyDataset(config['train_path'], Nch=[21], Rs=[80], Pch=[2],Nmodes=2,
                       window_size=window_size, strides=config['tbpl'], Nwindow=1000000, truncate=0,
                       Tx_window=True, pre_transform='Rx')

test_data = MyDataset(config['test_path'], Nch=[21], Rs=[80], Pch=[2],Nmodes=2,
                      window_size=net.overlaps + tbpl*20, strides=config['tbpl'], Nwindow=batch_size, truncate=0,
                      Tx_window=True, pre_transform='Rx')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


writer = SummaryWriter('_outputs/log_tensorboard/metadsp/train_dbp')
for epoch in range(epochs): 
    N = len(train_loader)
    train_loss = 0
    t0 = time.time()
    print('Train Loader batchs:', len(train_loader))
    for Rx, Tx, info in train_loader:
        
        signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
        signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
        info = info.to(device)

        y = net(signal_input, info)  # [B, L, N]
        y = conv(y)
        truth = signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
        loss = MSE(y.val, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(loss.item(), end='\r', flush=True)

    t1 = time.time()
    scheduler.step()
    res = test_model(net, conv, test_loader, device)

    writer.add_scalar('Loss/train', train_loss/N, epoch)
    writer.add_scalar('Loss/test', res['MSE'], epoch)
    writer.add_scalar('Metric/Qsq', res['Qsq'], epoch)
    writer.add_scalar('Metric/BER', res['BER'], epoch)
    print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
    print('Test BER: %.5f, Qsq: %.5f, MSE: %.5f' % (res['BER'], res['Qsq'], res['MSE']), flush=True)

    if epoch % 10 == 0:
        torch.save(net.state_dict(), f'_models/metadsp/train_dbp/net_{epoch}.pth')
        torch.save(conv.state_dict(), f'_models/metadsp/train_dbp/conv_{epoch}.pth')
        print('Model saved')