"""
Train FDBP in each mode.  Nmodes=1 or 2.
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
parser.add_argument('--config', type=str, default='configs/metadsp/fdbp_Nmodes1.yaml', help='path to config file')
args = parser.parse_args()
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def opt_pch(Rs, Nmodes=cfg['Nmodes']):
    if Nmodes == 1:
        if Rs == 40:
            return -4
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


with open(f'_outputs/paper_txt/FDBP_Nmodes{cfg["Nmodes"]}_{cfg["model_info"]["DBP_info"]["step"]}.txt','a') as f:
    f.write(f'################### FDBP step={cfg["model_info"]["DBP_info"]["step"]}: ################### \n')


log_path = '_outputs/log_tensorboard/A_MetaDSP_R1/' + f'FDBP_Nmodes{cfg["Nmodes"]}_{cfg["model_info"]["DBP_info"]["step"]}_{Rs}G_{Nch}ch_Pch{cfg["train_pch"]}'
model_path = '_models/A_MetaDSP_R1/' +  f'FDBP_Nmodes{cfg["Nmodes"]}_{cfg["model_info"]["DBP_info"]["step"]}_{Rs}G_{Nch}ch_Pch{cfg["train_pch"]}'

test_pch = opt_pch(80, Nmodes=cfg['Nmodes'])
test_info = {'Pch':test_pch, 'Rs': 80, 'Nch': 1, 'Nmodes':cfg['Nmodes'], "test_path": cfg['test_path']}

writer = SummaryWriter(log_path)
for key in cfg.keys():
    writer.add_text(key, str(cfg[key]))
writer.close()


net = LDBP(cfg['model_info']['DBP_info']).to(cfg['device'])
conv = downsamp(taps=64, Nmodes=cfg['Nmodes'], sps=2, init='zeros').to(cfg['device'])
net.to(cfg['device'])
conv.to(cfg['device'])

optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': cfg['lr']}, {'params': conv.parameters(), 'lr': cfg['lr']}])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_decay_step'], gamma=0.1)

train_data = MyDataset(cfg['train_path'], Nch=[Nch], Rs=[Rs], Pch=cfg['train_pch'], Nmodes=cfg['Nmodes'],
                window_size=cfg['tbpl'] + net.overlaps + conv.overlaps, strides=cfg['tbpl'], Nwindow=cfg['Nwindow'], truncate=0,
                Tx_window=True, pre_transform='Rx')
train_loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
print('Length of trainloader:', len(train_loader))

writer = Train(net, conv, train_loader, optimizer, scheduler, log_path, model_path, epoch_init=0, epochs=cfg['epochs'], test_info=test_info, save_log=True, save_model=True, save_interval=1, device=cfg['device'], model_info=cfg['model_info']['DBP_info'])

if cfg['Nmodes'] == 1:
    Qs = [Test(net, Pch=pch, Nch=Nch, Rs=Rs, Nmodes=cfg['Nmodes'])['Qsq'] for pch in range(-8,7)]
else:
    Qs = [Test(net, Pch=pch, Nch=Nch, Rs=Rs, Nmodes=cfg['Nmodes'])['Qsq'] for pch in range(-3,8)]

torch.save(Qs, f'_outputs/paper_Qp/FDBP_Nmodes{cfg["Nmodes"]}_{cfg["model_info"]["DBP_info"]["step"]}_{Rs}G_{Nch}ch_Pch{cfg["train_pch"]}.pt')

with open(f'_outputs/paper_txt/FDBP_Nmodes{cfg["Nmodes"]}_{cfg["model_info"]["DBP_info"]["step"]}.txt','a') as f:
    f.write(f'train on Pch={cfg["train_pch"]}, Rs={Rs}, Nch={Nch}, Qmax:'+ str(np.max(Qs)) + '\n')

writer.close()

