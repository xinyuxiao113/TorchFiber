"""
Generate Preprocessed data for PBC or NNeq.

python -m Jobs.generate_transformed_data
"""

import pickle , matplotlib.pyplot as plt, torch, numpy as np
from src.TorchDSP.dataloader import signal_dataset, get_data
from src.TorchDSP.baselines import CDCDSP, CDCtransform
from src.JaxSimulation.utils import show_symb
from src.TorchSimulation.receiver import  BER 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',   help='data file index. Final_batch2 or batch2', type=str, default='data/train_data_few.pkl')
parser.add_argument('--save_path',   help='save path', type=str, default='data/train_data_afterCDCDSP.pkl')
args = parser.parse_args()


data, info = get_data(args.data_path)
dataset = signal_dataset(data=data, batch_size=360,  shuffle=False)

# t:  P,Fi,Fs,Nch  
y,x,t = CDCDSP(dataset, device='cuda:0')
pickle.dump((y,x,t), open(args.save_path, 'wb'))