"""
Run baselines about optical fiber compensation.

python -m Jobs.baselines --data_path data/test_data.pkl --method CDC --Q_path Qfactor/largetest_MTLoss/CDC.pkl 
python -m Jobs.baselines --data_path data/test_data.pkl --method DBP --stps 1 --Q_path Qfactor/largetest_MTLoss/DBP_stps1.pkl 
"""
import torch, matplotlib.pyplot as plt, time, pickle, argparse
from src.TorchDSP.dataloader import signal_dataset, get_data
from src.TorchDSP.baselines import CDCDSP, DBPDSP
from src.TorchSimulation.receiver import  BER 


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',   help='data path',             type=str, default='data/test_data.pkl')
parser.add_argument('--Q_path',      help='path to save Q result', type=str, default='Qfactor/largetest_MTLoss/baselines.pkl')
parser.add_argument('--method',      help='CDC or DBP',            type=str, default='CDC')
parser.add_argument('--stps',        help='DBP steps per span',    type=int, default=10)
parser.add_argument('--batch_size',  help='batch_size',            type=int, default=360)
parser.add_argument('--lead_symbols',help='number of pilot symbol',type=int, default=2000)
args = parser.parse_args()

test_data, info = get_data(args.data_path)
test_dataset = signal_dataset(test_data, batch_size=args.batch_size, shuffle=False)

BER_batchsize = 357 if args.data_path == 'data/test_data.pkl' else 360
metric = {}


if args.method == 'CDC':
    predict, truth, task_info = CDCDSP(test_dataset, device='cuda:0', lead_symbols=args.lead_symbols)
    model_name = 'CDC'
elif args.method == 'DBP':
    predict, truth, task_info = DBPDSP(test_dataset, stps=args.stps, device='cuda:0', lead_symbols=args.lead_symbols)
    model_name = f'DBP stps={args.stps}'
else:
    raise ValueError('method should be CDC or DBP')


for n in [0, 1000, 10000, 20000]:
    res = BER(predict[:,n:].to('cpu'), truth[:,n:].to('cpu'), batch=BER_batchsize)
    metric[f'BER from {n}th symb'] = {model_name: res}


pickle.dump((metric, task_info), open(args.Q_path, 'wb'))