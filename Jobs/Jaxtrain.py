"""
Jax Model.
"""
import jax, optax, pickle, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
import sys 
from src.JaxSimulation.models import CDCBPS,CDCMIMO,DBPMIMO, DBPBPS, BER, piecewise_constant, DBP, mask_signal, mimoaf,downsamp, DBP_transform, wrap_signal, CDC_transform
from src.JaxSimulation.initializers import gauss, near_zeros, zeros, delta
import src.JaxSimulation.DataLoader as DL
from src.JaxSimulation.utils import  get_dtaps
import src.JaxSimulation.adaptive_filter as af
from src.JaxSimulation.train import Model
import matplotlib as mpl
from flax.core import freeze, unfreeze,lift
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['g',  'b', 'k', 'y', 'm', 'c']) # type: ignore

mark = 'Final_batch2'   # Nch=[1,3,5,7,9,11], Rs=[10,20,40,80,160], Power: [-8,-7,...,6], SF=1.2
SF = 1.2
mode = 1
device = 'cuda:0'
train_data, train_info = pickle.load(open('./data/train_data_few.pkl','rb'))


step = 5
ntaps = 401
dtaps = get_dtaps(step, train_data.a)
struct_info = {'module': ('DBP','MIMOAF'), 'lead_symbols':2000}
DBP_info = {'step':step, 'dtaps': dtaps,  'ntaps':ntaps, 'n_init':zeros, 'meta': 'scaler', 'batch_size':2000,
            'linear_kernel_type':'physical', 'linear_share':True, 'nonlinear_share':True,'optic_params_trainable': False, 
            'L':2000e3, 'D':16.5, 'Fc':299792458/1550E-9, 'gamma':0.0016567}

mimo_info = {'taps':32}

private_info = {'Fs':2*160*1e9, 'Fi':299792458/1550E-9, 'sps':2, 'Nmodes':1}

model_info = freeze({'struct_info':struct_info, 'DBP_info':DBP_info, 'mimo_info':mimo_info, 'private_info':private_info})

a = Model()
a.init(model_info=model_info,batch_dim=train_data.y.shape[0], update_state=True)
a.update_data(train_data, train_info, 2000)
a.update_optimizer(optax.experimental.split_real_and_imaginary(optax.adam(learning_rate=1e-4)), device=device)

for i in range(20):
    a.save(f'models/jax/MetaModel_full/model_MetaDBP_LMS_{i*100}')
    a.train(100)
a.save(f'models/jax/MetaModel_full/model_MetaDBP_LMS_2000')

