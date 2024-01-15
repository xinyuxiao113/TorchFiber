"""
Generate Preprocessed data for PBC or NNeq.

python -m Jobs.generate_transformed_data
"""

import pickle , matplotlib.pyplot as plt, torch, numpy as np
from src.TorchDSP.dataloader import signal_dataset, get_data
from src.TorchDSP.baselines import CDCDSP, CDCtransform
from src.JaxSimulation.utils import show_symb
from src.TorchSimulation.receiver import  BER 

train_data, train_info = get_data('data/train_data_few.pkl')
test_data, test_info = get_data('data/test_data_few.pkl')

train_dataset = signal_dataset(data=train_data, batch_size=360,  shuffle=False)
test_dataset = signal_dataset(data=test_data, batch_size=360,  shuffle=False)

# t:  P,Fi,Fs,Nch  
train_y, train_x,train_t = CDCtransform(train_dataset, device='cuda:0')
test_y, test_x,test_t = CDCtransform(test_dataset, device='cuda:0')

pickle.dump((train_y, train_x,train_t), open('data/train_data_afterCDC.pkl', 'wb'))
pickle.dump((test_y, test_x,test_t), open('data/test_data_afterCDC.pkl', 'wb'))


train_y, train_x,train_t = CDCDSP(train_dataset, device='cuda:0')
test_y, test_x,test_t = CDCDSP(test_dataset, device='cuda:0')

pickle.dump((train_y, train_x,train_t), open('data/train_data_afterCDCDSP.pkl', 'wb'))
pickle.dump((test_y, test_x,test_t), open('data/test_data_afterCDCDSP.pkl', 'wb'))