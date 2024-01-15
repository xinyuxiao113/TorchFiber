"""
Train MetaDSP Model.
"""
import pickle, torch, numpy as np, time
import argparse
import yaml
from torch.optim import Adam
from src.TorchDSP.core import TorchInput, TorchSignal, TorchTime
from src.TorchDSP.dsp import DSP
from src.TorchDSP.dataloader import signal_dataset, get_data

# Argument Parser
parser = argparse.ArgumentParser(description='Argument Parser for Your Program')
parser.add_argument('--config', help='Path to config.yaml file', type=str, default='config.yaml')
args = parser.parse_args()

# Load YAML configuration file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

device = 'cuda:0'
batch_size = config['batch_size']
epochs = config['epochs']
iters_per_batch = config['iters_per_batch']
tbpl = config['tbpl']                       # truncated backpropagation length
lr = config['lr']

# Load data
train_data, info = get_data(config['data_path'])

# Define model using configuration from YAML
DBP_info = config['DBP_info']
ADF_info = config['ADF_info']
net = DSP(DBP_info, ADF_info, batch_size=batch_size, mode='train')

# Training details
L = tbpl + net.overlaps
net = net.to(device)
optimizer = Adam(net.parameters(), lr=lr)
loss_list = []

def MTLoss(predict, truth):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=(-2,-1)))) 

def MeanLoss(predict, truth):
    return torch.log(torch.mean(torch.abs(predict- truth)**2))

loss_type = config['loss_type']
if loss_type == 'MT':
    loss_fn = MTLoss
elif loss_type == 'Mean':
    loss_fn = MeanLoss
else:
    raise ValueError('loss_type should be MT or Mean')

net = net.to(device)
checkpoint = {
    'model': net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'loss': loss_list,
    'epoch': epochs,
    'iters_per_batch': iters_per_batch,
    'tbpl': tbpl,
    'model info': {'DBP_info': DBP_info, 'ADF_info': ADF_info, 'batch_size': batch_size}
}
torch.save(checkpoint, config['model_path'] + f'/0.pth')

for epoch in range(epochs):
    dataset = signal_dataset(train_data, batch_size=batch_size, shuffle=True)
    for b, data in enumerate(dataset):
        net.adf.init_state(batch_size=data.signal_input.val.shape[0])
        net = net.to(device)
        for i in range(iters_per_batch):
            t0 = time.time()
            x = data.get_data(L, tbpl * i).to(device)
            y = net(x.signal_input, x.task_info, x.signal_output)  # [B, L, N]
            truth = x.signal_output.val[:, y.t.start:y.t.stop]     # [B, L, N]
            loss = loss_fn(y.val, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.adf.detach_state()

            t1 = time.time()

            if i % 1 == 0:
                print(f'Epoch {epoch} data batch {b}/{dataset.batch_number()} iter {i}/{iters_per_batch}: {loss.item()} time cost per iteration: {t1 - t0}', flush=True)
            loss_list.append(loss.item())

    checkpoint = {
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss_list,
        'loss_type': loss_type,
        'epoch': epochs + 1,
        'iters_per_batch': iters_per_batch,
        'tbpl': tbpl,
        'model info': {'DBP_info': DBP_info, 'ADF_info': ADF_info, 'batch_size': batch_size}
    }
    torch.save(checkpoint, config['model_path'] + f'/{epoch+1}.pth')