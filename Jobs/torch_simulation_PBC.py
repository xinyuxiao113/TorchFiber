import h5py, argparse
import numpy as np
import torch
from src.TorchDSP.train_pbc import  models, test_model
from src.TorchDSP.core import TorchSignal, TorchTime


parser = argparse.ArgumentParser(description="Simulation Configuration")
parser.add_argument('--path',   type=str, default='dataset/test.h5', help='dataset path')
args = parser.parse_args()

dic = torch.load('_models/Nmodes2_Nch3_Rs40/AMPBC_L400ckpt40', map_location='cpu')
net = models[dic['model_name']](**dic['model info'])
net.load_state_dict(dic['model'])
net.eval() 
net.cuda()


with h5py.File(args.path,'a') as f:
    for key in f.keys():
        if f[key].attrs['Nch'] == 3 and f[key].attrs['Rs'] == 40 and 'Rx_CDCDSP_PBC' not in f[key].keys():
            print('PBC testing for:', key)
            
            s = f[key]['Rx_CDCDSP'].attrs['start']
            e = f[key]['Rx_CDCDSP'].attrs['stop']
            Rx = torch.tensor(f[key]['Rx_CDCDSP'][...])
            Tx = torch.tensor(f[key]['Tx'][:,s:e])
            info = torch.tensor(f[key]['info'][:])

            signal = TorchSignal(val=Rx, t=TorchTime(0,0,1)).to('cuda:0')

            tbpl = 5000
            Ls = net.overlaps + tbpl
            start_idx = 0
            pbcs = []

            while start_idx < signal.val.shape[1]:
                sig_len = min(Ls, signal.val.shape[1] - start_idx)
                inp = signal.get_slice(sig_len, start_idx)
                PBC = net(inp, info)
                start_idx += tbpl
                pbcs.append(PBC.val.cpu().data)
            pbcout = torch.cat(pbcs, dim=1)

            data = f[key].create_dataset('Rx_CDCDSP_PBC', data=pbcout.numpy())
            data.attrs['start'] = s + (net.overlaps//2)
            data.attrs['stop'] = e - (net.overlaps//2)
            data.attrs['sps'] = 1
            data.dims[0].label = 'batch'
            data.dims[1].label = 'time'
            data.dims[2].label = 'modes'

