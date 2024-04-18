import torch, h5py, time
from torch.utils.data import DataLoader, Dataset
from src.TorchDSP.core import TorchSignal, TorchInput, TorchTime
from src.TorchSimulation.receiver import BER
from src.TorchDSP.dataloader import  get_signals, opticDataset, MyDataset, get_k
import numpy as np
from src.TorchDSP.nneq import eqCNNBiLSTM
from src.TorchDSP.loss import BER_well, MSE, Qsq
from functools import partial

data = MyDataset(path='dataset/train.h5', Nch=3, Rs=40, Pch=2, window_size=41, transform='Rx_CDCDSP_PBC', Nsymb=1000000)
train_loader = DataLoader(data, batch_size=50000, shuffle=True, num_workers=16)

testdata = MyDataset(path='dataset/test.h5', Nch=3, Rs=40, Pch=2, window_size=41, transform='Rx_CDCDSP_PBC', Nsymb=200000)
test_loader = DataLoader(testdata, batch_size=50000, shuffle=True, num_workers=16)

net = eqCNNBiLSTM(M=41, Nmodes=2, channels=64, kernel_size=11, hidden_size=40, res_net=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)

epochs = 60
loss_fn = MSE
mus =  4*1.1**np.repeat(np.arange(0, epochs//2 + 1), 2)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def test_model(dataloader, model, loss_fn, device):
    model = model.to(device)
    model.eval()
    mse = 0 
    ber = 0
    power = 0

    N = len(dataloader)
    with torch.no_grad():
        for x, y, z in dataloader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            y_pred = model(x)
            mse += loss_fn(y_pred, y).item()
            power += MSE(0, y).item()
            ber += BER(y, y_pred)['BER']
            
    return {'loss_fn': mse/N, 'BER': np.mean(ber/N), 'SNR': 10 * np.log10(power / mse), 'Qsq': Qsq(np.mean(ber/N)), 'BER_XY':ber/N, 'Qsq_XY': Qsq(ber/N)}


def train_model(train_loader, test_loader, net, loss_fn, optimizer, device, epochs=10, mus=[0]):
    net = net.to(device)
    metrics = []
    metrics.append(test_model(test_loader, net, loss_fn, device))

    for epoch in range(epochs + 1):
        net.train()
        train_loss = 0
        N = len(train_loader)

        loss_func = partial(loss_fn, mu=mus[epoch])
        t0 = time.time()
        for x,y,z in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            predict = net(x)
            loss = loss_func(predict, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(loss.item(), end='\r', flush=True)
        t1 = time.time()
        scheduler.step()

        print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
        metric = test_model(test_loader, net, loss_fn=MSE, device=device)
        print(metric, flush=True)
        metrics.append(metric)
    
    return metrics