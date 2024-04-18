"""
Train model.

"""
import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import StepLR
from functools import partial
import scipy.constants as const, scipy.special as special
from .dataloader import signal_dataset, get_k_batch, get_signals
from src.TorchSimulation.receiver import  BER
from .core import TorchSignal, TorchTime, dict_type
from .pbc import models as model_set1
from .nneq import models as model_set2
from .pbc_new import models_new as model_set3

models = {**model_set1, **model_set2, **model_set3}

optimizers = {
    "SGD": optim.SGD,
    "Adagrad": optim.Adagrad,
    "RMSprop": optim.RMSprop,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "ASGD": optim.ASGD,
    "LBFGS": optim.LBFGS,
    "SparseAdam": optim.SparseAdam
}

def Qsq(ber):
    return 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * ber), 0.))



def well(x, mu=4):
    return torch.sigmoid(mu*(x - 0.3162)) + torch.sigmoid(mu*(-x - 0.3162))

def BER_well(predict, truth, weight=None, mu=1):
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))

# 4-order loss: we hope the model concentrate on the large error signal.  (reduce the gap of loss function and BER)
# when the error is small, we should not learn the error, because the error is purely guasisan noise.
mus = 10**(np.linspace(-3, np.log10(0.31622776), 200))

def loss3(predict, truth, weight=None, mu=0.05):
    return torch.mean(torch.max(torch.tensor(mu), torch.abs(predict - truth))**2)

def loss4(predict, truth, weight=None, mu=0):
    return torch.mean(torch.abs(predict - truth)**4)


# Multi-Task loss
def MTLoss(predict, truth, weight=None, mu=0):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=(-2,-1)))) 

# log-MSE loss
def MeanLoss(predict, truth, weight=None, mu=0):
    return torch.log(torch.mean(torch.abs(predict- truth)**2))

# MSE loss
def MSE(predict, truth, weight=None, mu=0):
    return torch.mean(torch.abs(predict- truth)**2)

# weighted loss  predict: [B, W, Nmodes], weight: [B]
def weightedMSE(predict, truth, weight, mu=0):
    return torch.sum(torch.abs(predict- truth)**2 * weight[:,None, None])

# define L1 norm of model parameters
def L1(model, lamb=1e-4, device='cpu', mu=0) -> torch.Tensor:
    loss = torch.tensor(0., device=device)
    for param in model.parameters():
        loss += torch.norm(param, p=1)
    return lamb*loss


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

    # load history ckpt
    epoch1 = config['epochs']
    if 'ckpt_path' in config.keys(): config = torch.load(config['ckpt_path'])

    # data loading
    device = config['device']
    Pch = config['Pch'] if 'Pch' in config.keys() else None
    idx_train = config['idx_train'] if 'idx_train' in config.keys() else (0, None)
    idx_test = config['idx_test'] if 'idx_test' in config.keys() else (0, None)
    train_signal, train_truth, train_z = get_signals(config['train_path'], config['Nch'], config['Rs'], Pch=Pch,  device='cpu', idx=idx_train)
    test_signal, test_truth, test_z = get_signals(config['test_path'], config['Nch'], config['Rs'], device='cpu', idx=idx_test)

    # define model and load model
    model = models[config['model_name']]
    net = model(**config['model info'])
    if 'model' in config.keys(): net.load_state_dict(config['model'])
    net = net.to(device)
    net.train()

    # define optimizer
    if config['opt'] in optimizers.keys(): 
        if 'opt info' in config.keys():
            optimizer = optimizers[config['opt']](filter(lambda p: p.requires_grad, net.parameters()), **config['opt info'])
        else:
            optimizer = optimizers[config['opt']](filter(lambda p: p.requires_grad, net.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
    else: raise ValueError('optimizer not found')
    if 'optimizer' in config.keys(): optimizer.load_state_dict(config['optimizer'])

    # define schedule
    if "scheduler" in config.keys():
        scheduler = optim.lr_scheduler.__dict__[config['scheduler']](optimizer, **config['scheduler info'])
    else:
        scheduler = StepLR(optimizer, step_size=50, gamma=1)  # 每10个epoch，将学习率缩小为原来的1倍

    # define batch size
    Ls = net.overlaps + config['tbpl']
    train_loss_list = []
    val_loss_list = []

    # define loss function
    if config['loss_type'] == 'MT': loss_fn = MTLoss
    elif config['loss_type'] == 'Mean': loss_fn = MeanLoss
    elif config['loss_type'] == 'MSE': loss_fn = MSE
    elif config['loss_type'] == 'WMSE': loss_fn = weightedMSE
    elif config['loss_type'] == 'loss3': loss_fn = loss3
    elif config['loss_type'] == 'loss4': loss_fn = loss4
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

    epoch0 = config['train epoch'] + 1 if 'train epoch' in config.keys() else 1

    for epoch in range(epoch0, epoch1 + 1):
        train_loss = 0
        weight = 1
        loss_func = partial(loss_fn, mu=mus[epoch])
        for i in range(config['batchs']):
            # The ith batch training data
            signal = train_signal.get_slice(Ls, config['tbpl']*i + config['train_discard']).to(device)
            truth = train_truth.get_slice(Ls, config['tbpl']*i + config['train_discard']).to(device)
            train_z = train_z.to(device)
            P = 10**(train_z[:,0]/10)   # Power
            weight = P / torch.sum(P)
            for j in range(config['iters_per_batch']):
                t0 = time.time()
                predict = net(signal, train_z)
                L1_loss = L1(net, config['lamb'], device=device)
                fit_loss = loss_func(predict.val, truth.val[...,predict.t.start:predict.t.stop,:], weight)
                loss = fit_loss + L1_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())
                t1 = time.time()
                print(f'Epoch {epoch} Batch {i} iter {j}: total train loss: {loss.item()} fit loss: {fit_loss.item()}, L1 loss: {L1_loss.item()} time cost per iteration: {t1-t0}', flush=True)
            train_loss += fit_loss.item() # type: ignore
            writer.add_scalar('Loss/batch_train_loss', fit_loss.item(), (epoch-1)*config['batchs'] + i)  # type: ignore
            writer.add_scalar('Loss/batch_L1_loss', L1_loss.item(), (epoch-1)*config['batchs'] + i)      # type: ignore

        scheduler.step()

        print('\n' + '#' * 20 + '\n', flush=True)
        train_loss /= config['batchs']
        train_loss_list.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        print(f'Epoch {epoch} train loss: {train_loss}', flush=True)

        # validation data
        signal = train_signal.get_slice(Ls, config['tbpl']*config['batchs'] + config['train_discard']).to(device)  
        truth = train_truth.get_slice(Ls, config['tbpl']*config['batchs'] + config['train_discard']).to(device)
        predict = net(signal, train_z)
        val_loss = loss_func(predict.val, truth.val[...,predict.t.start:predict.t.stop,:], weight)
        val_loss_list.append(val_loss.item())

        # write to tensorboard and print
        writer.add_scalar('Loss/val', val_loss.item(), epoch)
        print(f"Epoch [{epoch}/{config['epochs']}], val Loss: {val_loss:.4f}")
        del signal, truth, predict
        torch.cuda.empty_cache()
        
        show_interval = config['show_interval'] if 'show_interval' in config.keys() else config['save_interval']

        checkpoint = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss_list,
                'val_loss': val_loss_list,
                'train epoch': epoch,
                **config, 
            }
        
        if (epoch) % show_interval == 0:
            # test model
            y_, x_ = test_model(checkpoint, test_signal, test_truth, test_z, device=device)
            metric = calculate_BER(y_, x_, config['ber_discard'])
            checkpoint['metric'] = metric
            y_,x_ = 0,0 
            torch.cuda.empty_cache()
            writer.add_scalar('Metric/BER_min', np.min(np.mean(metric['BER'],axis=-1)), epoch)
            writer.add_scalar('Metric/Q_max', Qsq(np.min(np.mean(metric['BER'],axis=-1))), epoch)
            writer.add_scalar('Metric/BER_mean', np.mean(metric['BER']), epoch)
            writer.add_scalar('Metric/Q_mean', Qsq(np.mean(metric['BER'])), epoch)
            print(f"Epoch [{epoch}/{config['epochs']}] BER: ", metric['BER'], flush=True)
            print(f"Epoch [{epoch}/{config['epochs']}] Qsq: ", metric['Qsq'], flush=True)

        if epoch % config['save_interval'] == 0:
            torch.save(checkpoint, config['model_path'] + f'ckpt{epoch}')
            print(f"Epoch [{epoch}/{config['epochs']}] Model saved!", flush=True)
        print('\n' + '#' * 20 + '\n', flush=True)
    writer.close()




def test_model(ckpt: dict, test_signal: TorchSignal, test_truth: TorchSignal, test_z: torch.Tensor, device='cuda:0'):
    '''
    Test model with ckpt. return predict signal and truth signal.

    Input:
        ckpt: checkpoint, type: dict
        test_signal: test signal, type: TorchSignal
        test_truth: test truth, type: TorchSignal
        test_z: test z, type: np.ndarray
    
    Output:
        predict, truth 
        predict: tensor with shape (batch, L, Nmodes)
        truth: tensor with shape (batch, L, Nmodes)
    '''
    test_signal = test_signal.to(device)
    test_truth = test_truth.to(device)
    test_z = test_z.to(device)
    net = models[ckpt['model_name']](**ckpt['model info'])
    net.load_state_dict(ckpt['model'])
    net = net.to(device)
    net.eval()
    Ls = net.overlaps + ckpt['tbpl']

    y_ = []
    x_ = []
    print(f'Testing: ', flush=True)
    t0 = time.time()
    batchs = -(-test_signal.val.shape[-2] // ckpt['tbpl'])
    for i in range(batchs):
        length = min(Ls, test_signal.val.shape[-2] - i*ckpt['tbpl'])
        signal = test_signal.get_slice(length, i*ckpt['tbpl'])
        truth = test_truth.get_slice(length, i*ckpt['tbpl'])
        with torch.no_grad():
            pbcout = net(signal, test_z)
            truth = truth.val[...,pbcout.t.start:pbcout.t.stop,:]
            y_.append(pbcout.val.data.to('cpu'))
            x_.append(truth.to('cpu'))
    t1 = time.time()
    print('Time cost:', t1-t0, flush=True)

    return torch.cat(y_, dim=1),  torch.cat(x_, dim=1)


def calculate_BER(y, x, n):
    '''
    Calculate BER discard n symbols.
    '''
    print('Calculate BER discard %d symbols' % n, flush=True)
    t0 = time.time()
    metric = BER(y[...,n:,:], x[...,n:,:])
    t1 = time.time()
    print('Time for each BER calculation: ', t1-t0, flush=True)
    return metric