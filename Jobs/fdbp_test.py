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


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/metadsp/fdbp.yaml', help='path to config file')
parser.add_argument('--job_name', type=str, default='test', help='job name')
args = parser.parse_args()

def opt_pch(Rs):
    if Rs == 40:
        return -4
    elif Rs == 80:
        return 0
    elif Rs == 160:
        return 1


for Rs in [160]:
    for Nch in [11]:
        
        log_path = '_outputs/log_tensorboard/train_dbp/' + 'train_{Rs}G_{Nch}ch_FDBP'.format(Rs=Rs, Nch=Nch)
        model_path = '_models/train_dbp/' + 'train_{Rs}G_{Nch}ch_FDBP'.format(Rs=Rs, Nch=Nch)
        writer = SummaryWriter(log_path)
        test_info = {'Pch':opt_pch(Rs), 'Rs': Rs, 'Nch': Nch, 'Nmodes':1, "test_path": "dataset_A800/test.h5"}

        with open(args.config) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        for key in cfg.keys():
            writer.add_text(key, str(cfg[key]))
        writer.add_text('job_name', args.job_name)
        writer.close()


        net = LDBP(cfg['model_info']['DBP_info']).to(cfg['device'])
        conv = downsamp(taps=64, Nmodes=cfg['Nmodes'], sps=2, init='zeros').to(cfg['device'])
        net.to(cfg['device'])
        conv.to(cfg['device'])

        optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': cfg['lr']}, {'params': conv.parameters(), 'lr': cfg['lr']}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_data = MyDataset(cfg['train_path'], Nch=[Nch], Rs=[Rs], Pch=cfg['train_pch'], Nmodes=cfg['Nmodes'],
                        window_size=cfg['tbpl'] + net.overlaps + conv.overlaps, strides=cfg['tbpl'], Nwindow=cfg['Nwindow'], truncate=0,
                        Tx_window=True, pre_transform='Rx')
        train_loader = DataLoader(train_data, batch_size=10, shuffle=True, drop_last=True)
        print(len(train_loader))

        writer = Train(net, conv, train_loader, optimizer, scheduler, log_path, model_path, epoch_init=0, epochs=cfg['epochs'], test_info=test_info, save_log=True, save_model=True, save_interval=1, device=cfg['device'], model_info=cfg['model_info']['DBP_info'])
        

        Qs = [Test(net, Pch=pch, Nch=Nch, Rs=Rs, Nmodes=1)['Qsq'] for pch in range(-8,7)]
        with open('_outputs/paper_result/qs.txt','a') as f:
            f.write(f'Rs={Rs}, Nch={Nch}, Qmax:'+ str(np.max(Qs)) + '\n')
        
        writer.close()

