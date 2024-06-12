"""
Train MetaDSP Model with split windows.
"""

import pickle, torch, numpy as np, time, argparse, os , yaml, jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from functools import partial
from torch.utils.data import DataLoader
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
parser.add_argument('--config', type=str, default='configs/metadsp/dbp_info.yaml', help='path to config file')
parser.add_argument('--shuffle', type=bool, default=True, help='shuffle data')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def opt_pch(Rs, Nmodes=cfg['Nmodes']):
    if Nmodes == 1:
        if Rs == 40:
            return -5
        elif Rs == 80:
            return 0
        elif Rs == 160:
            return 1
    elif Nmodes == 2:
        if Rs == 40:
            return -1
        elif Rs == 80:
            return 2
        elif Rs == 160:
            return 4
        
with open(f'_outputs/paper_txt/DBP_Nmodes{cfg["Nmodes"]}.txt','a') as f:
    f.write(f'################### DBP step={cfg["DBP_info"]["step"]}: ################### \n')

for Rs in cfg['train_rs']:
    for Nch in cfg['train_nch']:  
        print(f'############################ Rs={Rs}, Nch={Nch}:  ############################')
        log_path = '_outputs/log_tensorboard/A_MetaDSP_R1/' + 'DBP_{Rs}G_{Nch}ch_Pch{Pch}_Nmodes{Nmodes}'.format(Rs=Rs, Nch=Nch,Pch=cfg['train_pch'], Nmodes=cfg['Nmodes'])
        model_path = '_models/A_MetaDSP_R1/' + 'DBP_{Rs}G_{Nch}ch_Pch{Pch}_Nmodes{Nmodes}'.format(Rs=Rs, Nch=Nch,Pch=cfg['train_pch'], Nmodes=cfg['Nmodes'])
 
        writer = SummaryWriter(log_path)
        for key in cfg.keys():
            writer.add_text(key, str(cfg[key]))
        writer.close()

        net = TestDBP(**cfg['DBP_info'], Fs=Rs*2e9)
        conv = downsamp(taps=cfg['conv_taps'], Nmodes=cfg['DBP_info']['Nmodes'], sps=2, init='zeros')
        net.to(cfg['device'])
        conv.to(cfg['device'])

        optimizer = torch.optim.Adam([{'params': net.Dkernel_real, 'lr': cfg['D_lr']},{'params': net.Dkernel_imag, 'lr': cfg['D_lr']}, {'params': net.Nkernel, 'lr': cfg['N_lr']}, {'params': conv.parameters(), 'lr': cfg['conv_lr']}])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['scheduler_step'], gamma=cfg['scheduler_decay'])

        train_data = MyDataset(cfg['train_path'], Nch=[Nch], Rs=[Rs], Pch=cfg['train_pch'], Nmodes=cfg['Nmodes'],
                            window_size=cfg['tbpl'] + net.overlaps + conv.overlaps, strides=cfg['tbpl'], 
                            Nwindow=cfg['Nwindow'], truncate=0, Tx_window=True, pre_transform='Rx')
        train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
        print('data batchs:', len(train_loader))


        test_info = {'Nch': Nch, 'Rs': Rs, 'Pch':opt_pch(Rs, cfg['Nmodes']), 'Nmodes':cfg['Nmodes']}
        writer = Train(net, conv, train_loader, optimizer, scheduler, log_path, model_path, 0,  cfg['epochs'], test_info=test_info,  save_log=True, save_model=True, save_interval=10, device=cfg['device'], model_info=cfg['DBP_info'])


        if cfg['Nmodes'] == 1:
            Pchs = range(-8, 7)
        else:
            Pchs = range(-3, 8)
        Qs = [Test(net, Pch=pch, Nch=Nch, Rs=Rs, Nmodes=cfg['Nmodes'], test_path=cfg['test_path'], ber_discard=20000)['Qsq'] for pch in Pchs]
        print('Q factor:', Qs)
        
        torch.save(Qs, f'_outputs/paper_Qp/DBP_Nmodes{cfg["Nmodes"]}_Rs{Rs}_Nch{Nch}.pt')
        with open(f'_outputs/paper_txt/DBP_Nmodes{cfg["Nmodes"]}.txt','a') as f:
            f.write(f'train on Pch={cfg["train_pch"]}, Rs={Rs}, Nch={Nch}, Qmax:'+ str(np.max(Qs)) + '\n')
        
        writer = SummaryWriter(log_path)
        for i, q in enumerate(Qs):
            writer.add_scalar(f'Qp({Pchs[0]} : {Pchs[-1]+1} dBm)', q, i)
        writer.close()



