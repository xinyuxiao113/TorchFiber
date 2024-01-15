"""
Test PBC model on seperated data. (different Rs, Nch)

python -m TorchDSP.test_pbc
"""
import torch, pickle, matplotlib.pyplot as plt, argparse, time
from src.TorchDSP.core import TorchSignal,TorchTime
from src.TorchDSP.dataloader import get_signals, get_k_batch
from src.TorchDSP.train_pbc import MTLoss, MeanLoss, MSE, L1, test_model, calculate_BER, train_model
from src.TorchSimulation.receiver import  BER
from src.TorchDSP.pbc import models as model_set1
from src.TorchDSP.nneq import models as model_set2

models = {**model_set1, **model_set2}


parser = argparse.ArgumentParser()
# data path and Q path
parser.add_argument('--test_path', help='Test data path: data/test_data_afterCDCDSP.pkl.', type=str,  default='data/test_data_afterCDCDSP.pkl')
parser.add_argument('--Q_path',    help='Q save path: Qfactor_1205/Q2000/few/HPBC_L100_rho0.5_steps1_lossMT_batchs4_tbpl2e4_iters100_lr1e-4_epoch1.pkl', type=str,  default='Qfactor_1205/Q2000/few/HPBC_L100_rho0.5_steps1_lossMT_batchs4_tbpl2e4_iters100_lr1e-4_epoch1.pkl')
parser.add_argument('--model_path_dir',   help='model path directory. Default: models/HPBC/L100_rho0.5_steps1_lossMT_batchs4_tbpl2e4_iters100_lr1e-4_epoch1', type=str, default='models/HPBC/L100_rho0.5_steps1_lossMT_batchs4_tbpl2e4_iters100_lr1e-4_epoch1')

parser.add_argument('--BER_n',        help='Calculate BER after discard n symbols.',  type=int,   nargs='+', default=[0, 500, 1000, 1500, 2000, 10000])
parser.add_argument('--Nch',    help='Choose number of channels',      type=int,   nargs='+', default=[1,3,5,7,9,11])
parser.add_argument('--Rs',     help='Choose single channel symbol rate. [Hz]',  type=int,   nargs='+', default=[20,40,80,160])
args = parser.parse_args()

test_y, test_x,test_t = pickle.load(open(args.test_path, 'rb'))
res = {}
y = []
x = []
info = []

paths = []
for Rs in args.Rs:
    for Nch in args.Nch:
        paths.append(args.model_path_dir + '/' + f'Nch{Nch}_Rs{Rs}.pth')

for path in paths:
    dic = torch.load(path)
    test_signal, test_truth, test_z = get_signals(args.test_path, dic['Nch'], dic['Rs'], device='cuda:0')
    info.append(test_z)
    y_, x_ = test_model(dic, test_signal, test_truth, test_z)
    y.append(y_)
    x.append(x_)

y = torch.cat(y, dim=0)
x = torch.cat(x, dim=0)
info = torch.cat(info, dim=0)

for n in args.BER_n:
    metric = calculate_BER(y, x, n)
    res[f'BER from {n}th symb'] = {args.model_path_dir: metric}

pickle.dump((res, info), open(args.Q_path, 'wb'))