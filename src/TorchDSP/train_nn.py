import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os
from functools import partial
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch, numpy as np
import scipy.constants as const, scipy.special as special
from src.TorchSimulation.receiver import  BER
from .dataloader import  get_signals, opticDataset
from .nneq import models
from .opt import optimizers
from .loss import BER_well, MSE, SNR, Qsq



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



def train_model(config: dict):
    '''
        Train model with config  or ckpt.
        checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss_list,
                'val_loss': val_loss_list,
                'train epoch': epoch0 + (epoch + 1),
                **config, 
            }
    '''
    if 'seed' in config.keys():
        torch.manual_seed(config['seed'])

    # create path
    writer = SummaryWriter(config['tensorboard_path'])
    folder_path = os.path.dirname(config['model_path'])
    if not os.path.exists(folder_path): os.makedirs(folder_path)

    # data loading
    device = config['device']
    Pch = None if 'Pch' not in config.keys() else config['Pch']
    trainset = opticDataset(Nch=config['Nch'], Rs=config['Rs'], M=config['Memory'], Pch=Pch, path=config['train_path'], idx=config['idx_train'], power_fix=False)
    testset = opticDataset(Nch=config['Nch'], Rs=config['Rs'], M=config['Memory'], Pch=Pch, path=config['test_path'], idx=config['idx_test'], power_fix=False)
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

    # define model and load model
    model = models[config['model_name']]
    net = model(**config['model info'])
    net = net.to(device)
    net.train()

    if config['model_name'] == 'eqAMPBCaddNN' and 'NN_ckpt' in  config.keys():
        checkpoint = torch.load(config['NN_ckpt'])
        net.nn.load_state_dict(checkpoint['model'])
        print('NN loaded')

    # define optimizer
    if config['opt'] == 'AlterOptimizer':
        assert config['model_name'] == 'eqAMPBCaddNN', 'AlterOptimizer only support eqAMPBCaddNN'
        alternate = False if 'alternate' not in config.keys() else config['opt info']['alternate']
        optimizer = optimizers[config['opt']]([net.pbc.parameters(), net.nn.parameters()], config['opt info']['lr_list'], alternate=alternate)
    elif config['opt'] in optimizers.keys(): 
        if 'opt info' in config.keys():
            optimizer = optimizers[config['opt']](filter(lambda p: p.requires_grad, net.parameters()), **config['opt info'])
        else:
            optimizer = optimizers[config['opt']](filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
    else: raise ValueError('optimizer not found')

    # define schedule
    if "scheduler" in config.keys():
        scheduler = optim.lr_scheduler.__dict__[config['scheduler']](optimizer, **config['scheduler info'])
    else:
        scheduler = StepLR(optimizer, step_size=50, gamma=1)  # 每10个epoch，将学习率缩小为原来的1倍

    # define loss function and aneal strategy.
    if config['loss_type'] == 'MSE': loss_fn = MSE
    elif config['loss_type'] == 'BER_well': loss_fn = BER_well
    else: raise ValueError('loss_type should be MT or Mean')
    
    if 'adapt_mu' in config.keys() and config['adapt_mu'] == True and config['loss_type'] == 'BER_well':
        mus = 4*1.1**np.repeat(np.arange(0, config['epochs']//2 + 1), 2)
    elif 'adapt_mu' in config.keys() and config['adapt_mu'] == True and config['loss_type'] == 'MSE':
        mus = 10**(np.linspace(-3, np.log10(0.31622776), config['epochs'] + 1))
    elif 'adapt_mu' in config.keys() and config['adapt_mu'] == False and config['loss_type'] == 'BER_well':
        raise ValueError('adapt_mu should be True when loss_type is BER_well')
    else:
        mus = np.zeros(config['epochs'] + 1)

    for epoch in range(config['epochs'] + 1):
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
        
        writer.add_scalar('Loss/train', train_loss/N, epoch)
        writer.add_scalar('Loss/test', metric['loss_fn'], epoch)
        writer.add_scalar('Metric/Qsq', metric['Qsq'], epoch)
        writer.add_scalar('Metric/SNR', metric['SNR'], epoch)
        writer.add_scalar('Metric/BER', metric['BER'], epoch)
        writer.add_scalar('Metric/Qsq_X', metric['Qsq_XY'][0], epoch)
        writer.add_scalar('Metric/Qsq_Y', metric['Qsq_XY'][1], epoch)
        

        if epoch % config['save_interval'] == 0:
            checkpoint = {
                'model': net.state_dict(),
                'train epoch': epoch,
                'metric': metric,
                **config, 
            }
            torch.save(checkpoint, config['model_path'] + f'.ckpt{epoch}')
