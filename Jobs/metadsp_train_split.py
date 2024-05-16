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
from src.TorchDSP.dsp import DSP
from src.TorchDSP.dataloader import signal_dataset, get_data, MyDataset
from src.TorchDSP.loss import BER_well, MSE, SNR, Qsq
from src.TorchSimulation.receiver import  BER

# Argument Parser
parser = argparse.ArgumentParser(description='Argument Parser for Your Program')
parser.add_argument('--config', help='Path to config.yaml file', type=str, default='config.yaml')
args = parser.parse_args()
# Load YAML configuration file  
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

if 'model_path' not in config.keys(): config['model_path'] = args.config.replace('configs', '_models').replace('.yaml', '')
if 'tensorboard_path' not in config.keys(): config['tensorboard_path'] = args.config.replace('configs', '_outputs/log_tensorboard').replace('.yaml', '')


def MTLoss(predict, truth):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=(-2,-1)))) 

def MeanLoss(predict, truth):
    return torch.log(torch.mean(torch.abs(predict- truth)**2))


def Q_list(Rx, Tx, Ntest=10000):
    Q = []
    for t in  np.arange(0, Rx.shape[-2], 10000):
        Q.append(np.mean(BER(torch.from_numpy(Rx[t:t+Ntest]), torch.from_numpy(Tx[t:t+Ntest]))['Qsq']))
    return Q



def test_model(dataloader, ckpt):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for Rx, Tx, info in dataloader:
            Rx, Tx, info = Rx.to(device), Tx.to(device), info.to(device)
            signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
            signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
            y = net(signal_input, info, signal_output)  # [B, L, N]
            truth = signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
            break

    mse = MSE(y.val, truth).item()
    power = MSE(0, y.val).item()
    ber = BER(y.val, truth)['BER']
            
    return {'loss_fn': mse, 'BER': np.mean(ber), 'SNR': 10 * np.log10(power / mse), 'Qsq': Qsq(np.mean(ber))}




# Define model using configuration from YAML
net = DSP(**config['model_info'])
window_size = config['tbpl'] + net.overlaps


device = 'cuda:0'
batch_size = config['batch_size']
epochs = config['epochs']
tbpl = config['tbpl']                       # truncated backpropagation length
lr = config['lr']

# Load data
# train_data, info = get_data(config['train_path'])
train_data = MyDataset(config['train_path'], Nch=[21], Rs=[80], Pch=[2],Nmodes=2,
                       window_size=window_size, strides=config['tbpl'], Nwindow=1000000, truncate=0,
                       Tx_window=True, pre_transform='Rx')

test_data = MyDataset(config['test_path'], Nch=[21], Rs=[80], Pch=[2],Nmodes=2,
                      window_size=net.overlaps + tbpl*20, strides=config['tbpl'], Nwindow=batch_size, truncate=0,
                      Tx_window=True, pre_transform='Rx')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


net = net.to(device)
optimizer = Adam(net.parameters(), lr=lr)
schduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

loss_list = []

if config['loss_type'] == 'MT':
    loss_fn = MTLoss
elif config['loss_type'] == 'Mean':
    loss_fn = MeanLoss
elif config['loss_type'] == 'MSE':
    loss_fn = MSE
else:
    raise ValueError('loss_type should be MT or Mean')

net = net.to(device)
writer = SummaryWriter(config['tensorboard_path'])
if not os.path.exists(config['model_path']): os.makedirs(config['model_path'])

checkpoint = {
    'model': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss_list,
    'epoch': epochs,
    'tbpl': tbpl,
    'model info': config['model_info']
}
torch.save(checkpoint, config['model_path'] + f'/0.pth')

net.adf.init_state(batch_size=batch_size)
net = net.to(device)

for epoch in range(epochs):
    # net.adf.init_state(batch_size=batch_size)
    # net = net.to(device)
    N = len(train_loader)
    train_loss = 0
    t0 = time.time()
    print('Train Loader batchs:', len(train_loader))
    for Rx, Tx, info in train_loader:
        
        signal_input = TorchSignal(val=Rx, t=TorchTime(0,0,2)).to(device)
        signal_output = TorchSignal(val=Tx, t=TorchTime(0,0,1)).to(device)
        info = info.to(device)

        y = net(signal_input, info, signal_output)  # [B, L, N]
        truth = signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
        loss = loss_fn(y.val, truth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.adf.detach_state()

        train_loss += loss.item()
        print(loss.item(), end='\r', flush=True)

    t1 = time.time()
    schduler.step()

    print('Epoch: %d, Loss: %.5f, time: %.5f' % (epoch, train_loss/N, t1-t0), flush=True)
    metric = test_model(test_loader, net)
    print(metric, flush=True)

    writer.add_scalar('Loss/train', train_loss/N, epoch)
    writer.add_scalar('Loss/test', metric['loss_fn'], epoch)
    writer.add_scalar('Metric/Qsq', metric['Qsq'], epoch)
    writer.add_scalar('Metric/SNR', metric['SNR'], epoch)
    writer.add_scalar('Metric/BER', metric['BER'], epoch)


    if epoch % config['save_interval'] == 0:
        checkpoint = {
            'model': net.state_dict(),
            'model info': config['model_info'],
            'adf_state': net.adf.state,
            'optimizer': optimizer.state_dict(),
            'loss': loss_list,
            'loss_type': config['loss_type'],
            'epoch': epochs + 1,
            'tbpl': tbpl,
            
        }
        torch.save(checkpoint, config['model_path'] + f'/{epoch+1}.pth')