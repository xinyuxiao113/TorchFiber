"""
Train MetaDSP Model.   Old version.
"""
import pickle, torch, numpy as np, time
import argparse
from torch.optim import Adam
from src.TorchDSP.core import TorchInput, TorchSignal, TorchTime
from src.TorchDSP.dsp import DSP
from src.TorchDSP.dataloader import signal_dataset, get_data


parser = argparse.ArgumentParser(description='Argument Parser for Your Program')

# Data and Model Paths
parser.add_argument('--data_path',  help='Path to data. default data/train_data_few.pkl', type=str, default='data/train_data_few.pkl')
parser.add_argument('--model_path', help='Path to save model. default models/torch/net', type=str, default='models/torch/net')

# DBP and ADF Settings
parser.add_argument('--DBP', help='Choose DBP type: MetaDBP, FDBP.', type=str, default='MetaDBP')
parser.add_argument('--ADF', help='Choose ADF type: ddlms, lms, nlms, rmsprop, metalr, metaadam, metalstm,, metalstmtest, metalstmplus, metagru, metagrutest.', type=str, default='metatest')

# DBP details
parser.add_argument('--steps', help='Choose DBP steps:1,2,3,4,5. Default 5.', type=int, default=5)
parser.add_argument('--dtaps', help='Choose DBP dispersion kernel size. Default 5421.', type=int, default=5421)
parser.add_argument('--ntaps', help='Choose DBP nonlinear kernel size. Default 401.', type=int, default=401)

# MetaADF details
parser.add_argument('--Hdim', help='Hidden dimensions of RNN: 1,2,4,8,16.', type=int, default=16)
parser.add_argument('--Hdepth', help='Hidden depth of RNN: 1,2.', type=int, default=2)
parser.add_argument('--step_max', help='ADF step max: 5e-2, 1e-2.', type=float, default=5e-2)
parser.add_argument('--lr_init', help='MetaADF initial learning rates: (1/2**6,1/2**7) or ...', type=float, nargs=2, default=[1/2**6, 1/2**7])

# Training Parameters
parser.add_argument('--epochs', help='Number of training epochs: 10.', type=int, default=10)
parser.add_argument('--batch_size', help='Batch size: 90, 180, 360. default 360', type=int, default=360)
parser.add_argument('--iters_per_batch', help='Iterations per batch: 100, 200, 400.', type=int, default=200)
parser.add_argument('--tbpl', help='Truncation backpropagation length: 50, 100, 200. ', type=int, default=200)
parser.add_argument('--loss_type', help='Loss type: MT or Mean', type=str, default='MT')
parser.add_argument('--lr', help='Training learning rate: 1e-4, 1e-5', type=float, default=1e-4)

args = parser.parse_args()


device = 'cuda:0'
batch_size = args.batch_size
epochs = args.epochs
iters_per_batch = args.iters_per_batch
tbpl = args.tbpl                       # trucated backpropagation length
lr = args.lr

# load data
train_data, info = get_data(args.data_path)

# define model
DBP_info = {'step':args.steps, 'dtaps': args.dtaps,  'ntaps':args.ntaps, 'type': args.DBP, 'Nmodes':1,
            'L':2000e3, 'D':16.5, 'Fc':299792458/1550E-9, 'gamma':0.0016567,
            'task_dim':4, 'task_hidden_dim': 100}
meta_args = {'lr_init': tuple(args.lr_init)} if (args.ADF == 'ddlms' or args.ADF == 'metaadam' or args.ADF == 'lms' or args.ADF == 'nlms' or args.ADF == 'rmsprop') else {'step_max': args.step_max, 'hiddden_dim': args.Hdim, 'num_layers': args.Hdepth}
ADF_info = {'type':args.ADF ,'mimotaps': 32, 'Nmodes':1, 'meta_args':meta_args}
net = DSP(DBP_info, ADF_info, batch_size=batch_size, mode='train')

# training details
L = tbpl + net.overlaps
net = net.to(device)
optimizer = Adam(net.parameters(), lr=lr)
loss_list = []

def MTLoss(predict, truth):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=(-2,-1)))) 

def MeanLoss(predict, truth):
    return torch.log(torch.mean(torch.abs(predict- truth)**2))

if args.loss_type == 'MT':
    loss_fn = MTLoss
elif args.loss_type == 'Mean':
    loss_fn = MeanLoss
else:
    raise ValueError('loss_type should be MT or Mean')


net = net.to(device)
checkpoint = {'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss_list,
                'epoch': epochs,
                'iters_per_batch': iters_per_batch,
                'tbpl': tbpl,
                'model info': {'DBP_info': DBP_info, 'ADF_info': ADF_info,'batch_size': batch_size}
                }
torch.save(checkpoint, args.model_path + f'/0.pth')


for epoch in range(epochs):
    dataset = signal_dataset(train_data, batch_size=batch_size, shuffle=True)
    for b,data in enumerate(dataset):
        net.adf.init_state(batch_size=data.signal_input.val.shape[0])
        net = net.to(device)
        for i in range(iters_per_batch):
            t0 = time.time()
            x = data.get_data(L, args.tbpl*i).to(device)
            y = net(x.signal_input, x.task_info, x.signal_output)   # [B, L, N]
            truth = x.signal_output.val[:, y.t.start:y.t.stop]      # [B, L, N]
            loss = loss_fn(y.val, truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.adf.detach_state()

            t1 = time.time()

            if i % 1 == 0:
                print(f'Epoch {epoch} data batch {b}/{dataset.batch_number()} iter {i}/{iters_per_batch}:  {loss.item()}     time cost per iteration: {t1 - t0}', flush=True)
            loss_list.append(loss.item())

    checkpoint = {'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss_list,
                'loss_type': args.loss_type,
                'epoch': epochs,
                'iters_per_batch': iters_per_batch,
                'tbpl': tbpl,
                'model info': {'DBP_info': DBP_info, 'ADF_info': ADF_info,'batch_size': batch_size}
                }
    torch.save(checkpoint, args.model_path + f'/{epoch+1}.pth')
