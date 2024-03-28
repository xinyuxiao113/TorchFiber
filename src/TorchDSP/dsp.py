'''
    DSP module.
    DSP = DBP + ADF.
    DBP: Digital Back Propagation.
    ADF: Adaptive Decision Feedback.
    ADFCell: ADF cell.
    ADF: ADF module.
    LDBP: Learnable DBP module.
'''
import torch, copy, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F, torch.nn.init as init, torch.nn as nn
from typing import Union
from scipy import constants as const

from .core import TorchSignal, TorchTime
from .layers import MLP, ComplexConv1d, Parameter
from .utils import to_device, detach_tree, Dconv, Nconv, decision, constellation, tree_map
from .metaopt import MetaLr, MetaNone, MetaLSTMOpt, MetaLSTMtest, MetaLSTMplus, MetaAdam, NLMSOpt, RMSPropOpt, MetaGRUOpt, MetaGRUtest



class LDBP(nn.Module):
    '''
        DBP with hyper-network.
        
        Attributes:
            DBP_info: a dict of DBP parameters.
            method: string. 'FDBP' or 'MetaDBP'.
            dz: DBP step size. [m]
            dtaps: dispersion kernel size.
            ntaps: nonlinear filter size.
            Nmodes: signal polarization modes. 1 or 2.
            task_dim: task dimension. equal to 4.
            task_hidden_dim: hidden size used in MetaDBP.

        DBP_info example:
            DBP_info = {'step':5, 'dtaps': 5421,  'ntaps':401, 'type': args.DBP, 'Nmodes':1,
            'L':2000e3, 'D':16.5, 'Fc':299792458/1550E-9, 'gamma':0.0016567,
            'task_dim':4, 'task_hidden_dim': 100}

    '''
    def __init__(self, DBP_info: dict):
        super(LDBP, self).__init__()
        self.DBP_info = DBP_info
        self.method = DBP_info['type']
        self.dz = DBP_info['L'] / DBP_info['step']   # [m]
        self.dtaps = self.DBP_info['dtaps']
        self.ntaps = self.DBP_info['ntaps']
        self.Nmodes = DBP_info['Nmodes']
        self.task_dim = DBP_info['task_dim']
        self.task_hidden_dim = DBP_info['task_hidden_dim']

        if self.method == 'MetaDBP':
            self.task_mlp = MLP(self.task_dim, self.task_hidden_dim, self.ntaps * self.Nmodes**2)
        elif self.method == 'FDBP':
            self.task_mlp = Parameter(output_size = self.ntaps*self.Nmodes**2)
        else:
            raise(ValueError)
        self.gamma = self.DBP_info['gamma']                                                        # [1/W/m]
        

    def forward(self, signal: TorchSignal, task_info: torch.Tensor) -> TorchSignal:
        '''
            Input: 
                TorchSignal with val shape [B, L, Nmodes].
                task_info: torch.Tensor with shape [B, 4]. 
                task_info: [batch, 4],  [P, Fi, Fs, Nch]  Unit:[dBm, Hz, Hz, 1]

            Output:
                TorchSignal with val shape [B, L - C, Nmodes], where C = steps*(dtaps - 1 + ntaps - 1).
        '''
        x = signal.val  # [batch, L*sps, Nmodes]
        t = copy.deepcopy(signal.t)    # [start, stop, sps]
        batch = x.shape[0]
        beta2 = self.get_beta2(self.DBP_info['D'], self.DBP_info['Fc'])/1e3                   # [s^2/m]
        beta1 = self.get_beta1(self.DBP_info['D'], self.DBP_info['Fc'], task_info[:,1])/1e3   # [s/m]    (batch,)
        Dkernel = self.dispersion_kernel(-self.dz, self.dtaps, task_info[:,2], beta2, beta1, domain='time') # [batch, dtaps]   turn dz to negative.
        Nkernel = self.task_mlp(task_info).reshape(batch, self.Nmodes, self.Nmodes, self.ntaps)  # [batch, Nmodes, Nmodes, ntaps]
        P = 1e-3*10**(task_info[:,0]/10)/self.Nmodes                                        # [batch,]   [W]
        

        for i in range(self.DBP_info['step']):
            # Linear step
            x = Dconv(x, Dkernel, 1)              # [batch, L*sps - dtaps + 1, Nmodes]
            t.conv1d_t(self.dtaps, stride=1)

            # Nonlinear step
            start, stop = t.start, t.stop
            t.conv1d_t(self.ntaps, stride=1)
            phi = Nconv(torch.abs(x)**2, Nkernel, 1)  # [batch, L*sps - dtaps - ntaps + 2, Nmodes]   x: [batch, L*sps - dtaps + 1, Nmodes]   Nkernel: [batch, Nmodes, Nmodes, ntaps]
            x = x[:,t.start - start: t.stop - stop + x.shape[1]] * torch.exp(1j*phi * self.gamma * P[:,None,None] * self.dz)   # [batch, L*sps - dtaps + 1, Nmodes] turn dz to negative.
            
        return TorchSignal(x, t)

    def get_omega(self, Fs:torch.Tensor, Nfft:int) -> torch.Tensor:
        ''' 
        Get signal fft angular frequency.
        Input:
            Fs: sampling frequency. [Hz]          [batch,]
            Nfft: number of sampling points.      
        Output:
            omega: torch.Tensor [batch, Nfft]
        '''
        return 2*torch.pi*Fs[:,None] * torch.fft.fftfreq(Nfft)[None,:].to(Fs.device)  # [batch, Nfft]

    def get_beta1(self, D, Fc, Fi):
        '''
        Calculate beta1.
        Input:
            D:[ps/nm/km]    Fc: [Hz]   Fi: [Hz] 
        Output:
            beta1:    [s/km]
        '''
        beta2 = self.get_beta2(D, Fc)  # [s^2/km]
        beta1 = 2*np.pi * (Fi - Fc)*beta2 # [s/km]
        return beta1

    def get_beta2(self, D, Fc):
        '''
        Calculate beta2.
        Input:
            D:[ps/nm/km]    Fc: [Hz]
        Output:
            beta2:    [ps*s/nm]=[s^2/km]
        '''
        c_kms = const.c / 1e3                       # speed of light (vacuum) in km/s
        lamb  = c_kms / Fc                          # [km]
        beta2 = -(D*lamb**2)/(2*np.pi*c_kms)        # [ps*s/nm]=[s^2/km]
        return beta2

    
    def dispersion_kernel(self, dz:float, dtaps:int, Fs:torch.Tensor, beta2:float=-2.1044895291667417e-26, beta1: torch.Tensor=torch.zeros(()), domain='time') -> torch.Tensor:
        ''' 
        Dispersion kernel in time domain or frequency domain.
        Input:
            dz: Dispersion distance.              [m]
            dtaps: length of kernel.     
            Fs: Sampling rate of signal.          [Hz]
            beta2: 2 order  dispersion coeff.     [s^2/m]
            beta1: 1 order dispersion coeff.      [s/m]
            domain: 'time' or 'freq'
        Output:
            h:jnp.array. (dtaps,)
            h is symmetric: jnp.flip(h) = h.
        '''
        omega = self.get_omega(Fs, dtaps)      # (batch, dtaps)
        kernel = torch.exp(-1j*beta1[:,None]*omega*dz - 1j*(beta2/2)*(omega**2)*dz)  # beta1: (batch,)

        if domain == 'time':
            return torch.fft.fftshift(torch.fft.ifft(kernel, axis=-1), axis=-1)
        elif domain == 'freq':
            return kernel
        else:
            raise(ValueError)


    

class ADFCell(nn.Module):
    '''
        MetaADF Cell.

        Attributes:
            method: As string denote type of ADFCell takes value from 'lms', 'nlms', 'ddlms', 'metaadam',  'metalstm', 'metatest', 'metalstmplus'.
            taps: filter size.
            lead_symbols: number of pilot symbolsm, only work in test mode.
            Nmodes: polarozation modes. 1 or 2.
            grad_max: cut off value of gradient, ensure stableness of alg.
            const: constellation of transmmiter scheme.
            mode: 'train' or 'test'. In train mode, we use true symbol as refrence while decision symbol is considered in test mode.
            meta_args: a dict. meta parameters about metaadf.

    '''

    def __init__(self, method='meta', taps=32, Nmodes=1, grad_max=(30., 30.), lead_symbols=2000, mode='train', meta_args:dict={}):
        super(ADFCell, self).__init__()
        self.method = method
        self.taps = taps
        self.lead_symbols = lead_symbols
        self.Nmodes = Nmodes
        self.grad_max = grad_max
        self.const = constellation  # 常量张量
        self.mode = mode
        self.meta_args = meta_args
        self.normalized_grad = True   # defualt value
        self.setup_meta_optimizer()
    
    def setup_meta_optimizer(self):
        method_to_class = {
            'ddlms': MetaNone,
            'lms': MetaNone,
            'nlms': NLMSOpt,
            'rmsprop': RMSPropOpt,
            'metalr': MetaLr,
            'metalstm': MetaLSTMOpt,
            'metalstmtest': MetaLSTMtest,
            'metalstmplus': MetaLSTMplus,
            'metaadam': MetaAdam,
            'metagru': MetaGRUOpt,
            'metagrutest': MetaGRUtest
        }

        if self.method in method_to_class:
            self.MetaOpt = method_to_class[self.method](**self.meta_args)
            # 对于特定的方法，设置 normalized_grad 为 False
            if self.method in ['lms', 'nlms', 'rmsprop']:
                self.normalized_grad = False
        else:
            raise ValueError('method must be meta or filter')
    
    def forward(self, state, inp, task_info:Union[None, torch.Tensor]=None):
        '''
        state, inp  -> state, d
        '''
        # step 1: data, gradient
        u, x = inp
        grads, d, z, e, o = self.grad(state['theta'], inp, state['iter'])

        # Step 2: update hidden state (Meta learning rate)
        add_in = state['theta']

        # !TODO add method set
        if self.method == 'metalstmtest' or self.method == 'metalstmplus' or self.method == 'metagru' or self.method == 'metagrutest':
            more_info = (u, d,z,e, task_info)
            hidden, updates, lrs = self.MetaOpt(state['hidden'], grads, add_in, more_info)
        elif self.method == 'nlms':
            hidden, updates, lrs = self.MetaOpt(state['hidden'], grads, o)
        else:
            hidden, updates, lrs = self.MetaOpt(state['hidden'], grads, add_in)

        # Step 3: update parameters
        theta0 = state['theta']
        theta = tree_map(lambda x,y:x+y, theta0, updates)
        state['iter'] += 1
        return {'iter': state['iter'], 'theta': theta, 'hidden': hidden}, z, lrs
    
    def change_mode(self, mode):
        '''
        Change mode of ADFCell. 
            mode: 'train' or 'test'
        '''
        self.mode = mode
    
    def train(self, i):
        '''
        Return a bool variable indicate use truth symbol or not (decision-directed).
        '''
        if self.mode == 'train':
            return torch.tensor(True)
        elif self.mode == 'test':
            return i < self.lead_symbols
        else:
            raise(ValueError)

    def to(self,  *args, **kwargs):
        '''
        Move this whole module to ...(New device)
        '''
        super(ADFCell, self).to( *args, **kwargs)
        self.const = self.const.to(*args, **kwargs)
        return self

    def init_carry(self, batch,  dtype=torch.complex64):
        # w0 = torch.zeros(batch, self.Nmodes, self.Nmodes, self.taps, dtype=dtype)  # [batch, Nmodes, Nmodes, taps]
        w0 = self.gauss_init(self.taps, batch, self.Nmodes)                          # [batch, Nmodes, Nmodes, taps]
        f0 = torch.full((batch, self.Nmodes), 1., dtype=dtype)                       # [batch, Nmodes]
        theta = (w0, f0)
        hidden = self.MetaOpt.init_carry(theta)                                      # type: ignore # [batch, h]
        return {'iter': torch.zeros((), dtype=torch.int), 'theta': theta, 'hidden': hidden}

    
    def gauss_init(self, taps, batch, Nmodes, dtype=torch.complex64):
        y = torch.exp(-torch.linspace(-5,5, taps)**2)
        y = y / torch.sum(y)
        return y.repeat(batch, Nmodes, Nmodes, 1).to(dtype)


    def grad(self, theta:tuple, inp:tuple, i) -> tuple:
        '''
        Input:
            theta: (w, f) the adaptive weights of ADF.   w: [batch, Nmodes, Nmodes, taps]    f: [batch, Nmodes]
            inp: (u, x) input signal and truth signal. u: [batch, taps, Nmodes]    x: [batch, Nmodes]     
            i: current index of inp.
        Output:
            grads: (gw, gf) gradient w.r.t (w,f).  
            d: [batch, Nmodes]   
            k: [batch, Nmodes]   
            (e_w, e_f): error signal.
            (o1, v): additional info.
        '''
      
        u, x = inp
        w, f = theta
        v = self.mimo(w, u)  # [batch, Nmodes]
        k = v * f            # [batch, Nmodes]
        d = torch.where(self.train(i), x, decision(self.const, k))  # [batch, Nmodes]
        l = torch.sum(torch.abs(k - d)**2)
        Nmodes = u.shape[-1]

        psi_hat = torch.abs(f) / f    # [batch, Nmodes]
        e_w = d * psi_hat - v         # [batch, Nmodes]
        e_f = d - k                   # [batch, Nmodes]
        if self.normalized_grad:
            gw = -1. / ((torch.abs(u)**2).sum(dim=(1,2)) + 1e-9)[:, None, None, None] * e_w[..., None, None] * u.conj().permute([0, 2, 1])[:, None, :, :] # [batch, Nmodes, Nmodes, taps]
            gf = -1. / ((torch.abs(v)**2).sum(dim=(1,)) + 1e-9)[:, None] * e_f * v.conj()  # [batch, Nmodes]
            gf = torch.where(torch.abs(gf) > self.grad_max[0], gf / torch.abs(gf) * self.grad_max[0], gf)
        else:
            gw = -1. * e_w[..., None, None] * u.conj().permute([0, 2, 1])[:, None, :, :] # [batch, Nmodes, Nmodes, taps]
            gf = -1. * e_f * v.conj()  # [batch, Nmodes]
            gf = torch.where(torch.abs(gf) > self.grad_max[0], gf / torch.abs(gf) * self.grad_max[0], gf)

        o1 = torch.cat([u.conj().permute([0, 2, 1])[:,None,:,:]] * Nmodes, dim=1)
        return (gw, gf), d, k, (e_w, e_f), (o1,v)


    def apply_fn(self, theta, yf):
        '''
         ws: [batch, Nmodes, Nmodes, taps]    fs: [batch, Nmodes]        yf: [batch, Nmodes]
        '''
        ws, fs = theta
        return self.mimo(ws, yf) * fs 
    

    def mimo(self, w, u):
        '''
        [batch, Nmodes, Nmodes, taps] x [batch, taps, Nmodes] -> [batch, Nmodes]
        '''
        return torch.einsum('bijt,btj->bi', w, u)



class ADF(nn.Module):
    '''
        ADF with meta optimizer.

        Attributes:
            Cell: ADFCell with type=method. 'lms', 'nlms', 'ddlms', 'metaadam',  'metalstm', 'metatest', 'metalstmplus'.
            taps: filter size.
            Nmodes: polarization modes.
            batch_size: input batch size. (decision the carray state shape)
            lead_symbols: number of pilot symbols.
            mode: 'train' or 'test'.
            meta_args: meta parameters.
        
    '''
    def __init__(self, method='metalstm', taps=32, Nmodes=1, batch_size=64, lead_symbols=2000, mode='train', meta_args={}):
        super(ADF, self).__init__()
        self.taps = taps
        self.mode = mode
        self.Cell = ADFCell(method=method,taps=self.taps, Nmodes=Nmodes, mode=mode, lead_symbols=lead_symbols, meta_args=meta_args)
        self.batch_size = batch_size
        self.state = self.Cell.init_carry(batch_size)    # state:{'iter': torch.Size([]), 'theta': (torch.Size([batch, Nmodes, Nmodes, taps]), torch.Size([batch, Nmodes])), 'hidden': (torch.Size([num_layers, N * batch, 2*Co]), torch.Size([nunm_layers, N*batch, 2*Co]))}
    
    def forward(self, signal: TorchSignal, signal_pilot: TorchSignal, task_info: torch.Tensor, show_lr=False) -> Union[TorchSignal, list]:
        '''
        Input:
            signal: TorchSignal with val shape [batch, Nsymb*sps + taps - sps, Nmodes].
            signal_pilot: TorchSignal with val shape [batch, Nsymb, Nmodes].   
            task_info: torch.tensor with shape [batch, 4].
            show_lr: bool. True or False.
        Output:
            TorchSignal or list.
            show_lr=True: return list of lr in ADF.
            show_lr=False: return TorchSignal.
        '''
        assert signal.val.shape[0] == self.batch_size, 'change the batch_size to the right number with change_batchsize method.'
        x = signal.val
        t = copy.deepcopy(signal.t) 
        sps = t.sps
        t.conv1d_t(self.taps, sps)
        x = self.frame(x, self.taps, sps)                                             # x: [batch, Nsymb, taps, Nmodes]
        pilot = signal_pilot.val[:, t.start: signal_pilot.val.shape[1] + t.stop]      # signal_pilot: [batch, Nsymb, Nmodes]

        outs = []
        lrs = []
        for i in range(x.shape[1]):
            inp = (x[:,i,...], pilot[:,i,:])                                 # inp = (u, x),  u: [batch, taps, Nmodes]    x: [batch, Nmodes]
            self.state, outp, lr = self.Cell(self.state, inp, task_info)         # state, inp  -> state, d
            outs.append(outp)
            lrs.append(lr)
        x = torch.stack(outs, dim=1)  # [batch, Nsymb, Nmodes]

        if show_lr:
            return lrs
        else:
            return TorchSignal(x, t)
        
    def change_mode(self, mode:str):
        '''
        Change mode of ADF. 
            mode: 'train' or 'test'
        '''
        self.Cell.change_mode(mode)

    def change_lead_symbols(self, lead_symbols:int):
        '''
        Change lead_symbols of Cell. 
            mode: 'train' or 'test'
        '''
        self.Cell.lead_symbols = lead_symbols

    def change_batchsize(self, batch_size:int):
        '''
        Change batch size. 
            mode: 'train' or 'test'
        '''
        self.batch_size = batch_size
        self.state = self.Cell.init_carry(batch_size) 

    def to(self, device):
        '''
        Move this whole module to ...(New device)
        '''
        self.Cell = self.Cell.to(device)
        self.state = to_device(self.state, device) 
        return self
    
    def detach_state(self):
        '''
        Detach state from the calculation graph to prevent memory issue. (TBPLL method: train long RNN strategy)
        '''
        self.state = detach_tree(self.state)

    def init_state(self, batch_size:int=-1):
        '''
        Initialize state of Cell.
        '''
        if batch_size != -1:
            self.state = self.Cell.init_carry(batch_size) 
            self.batch_size = batch_size
        else:
            self.state = self.Cell.init_carry(self.batch_size) 

    def frame(self, x: torch.Tensor, taps: int, stride: int, fnum: Union[None, int]=None) -> torch.Tensor:
        '''
        Generate circular frame from Array x.
        Input:
            x: Arrays about to be framed with shape (B, L, dims)
            taps: frame length.
            stride: step size of frame.
            fnum: steps which frame moved. If fnum==None, then fnum --> 1 + (N - flen) // fstep
        Output:
            A extend array with shape (B, fnum, taps, dims)
        '''
        N = x.shape[1]
        if fnum == None:
            fnum = 1 + (N - taps) // stride
        
        ind = (torch.arange(taps)[None,:] + stride * torch.arange(fnum)[:,None]) % N
        return x[:,ind,...]
    


class downsamp(nn.Module):
    '''
    Downsample module.
    A simple replacement of ADF.
    '''
    def __init__(self, taps=32, Nmodes=1, batch_size=None, sps=2):
        super(downsamp, self).__init__()
        self.taps = taps
        self.Nmodes = Nmodes
        self.conv = ComplexConv1d(Nmodes, Nmodes, self.taps, stride=sps, padding=0, bias=False)

    def forward(self, signal: TorchSignal):
        x = signal.val                     # [batch, L*sps, Nmodes]
        sps = signal.t.sps
        t = copy.deepcopy(signal.t)
        t.conv1d_t(self.taps, sps)
        y = self.conv(x.permute([0,2,1]))  # [batch, Nmodes, L - taps + 1]
        y = y.permute([0,2,1])             # [batch, L - taps + 1, Nmodes]
        return TorchSignal(y, t)

    def detach_state(self):
        pass


class DSP(nn.Module):
    '''
        DSP module.  DSP = DBP + ADF.
        Initialize:
            DBP_info: DBP information dict.
            ADF_info: ADF information dict.
            batch_size: int.
            mode: 'train' or 'test'.
        Attributes:
            mode: 'train' or 'test'.
            ldbp: LDBP module.
            adf: ADF module.
            overlaps: overlaps of the whole pipeline. Every convolution operator will cause overlaps, so we should consider this issue.

        Example:
            DBP_info = {'step':5, 'dtaps': 5421,  'ntaps':401, 'type': 'MetaDBP', 'Nmodes':1,
                    'L':2000e3, 'D':16.5, 'Fc':299792458/1550E-9, 'gamma':0.0016567,
                    'task_dim':4, 'task_hidden_dim': 100}
            ADF_info = {'type':'metalstm' ,'mimotaps': 32, 'Nmodes':2, 
                    'meta_args': {'step_max': 1e-2, 'in_dim': 2, 'hiddden_dim': 16, 'num_layers': 2}}
            net = DSP(DBP_info, ADF_info, batch_size=train_data.task_info.shape[0])
    '''
    def __init__(self, DBP_info: dict, ADF_info: dict, batch_size=8, mode:str='train'):
        super(DSP, self).__init__()
        self.mode = mode
        self.ldbp = LDBP(DBP_info)
        self.adf = ADF(method=ADF_info['type'], taps=ADF_info['mimotaps'], Nmodes=ADF_info['Nmodes'], batch_size=batch_size, mode=mode, meta_args=ADF_info['meta_args'])
        self.overlaps = DBP_info['step'] * ((DBP_info['dtaps'] - 1) + (DBP_info['ntaps'] - 1)) // 2 + (ADF_info['mimotaps']-1)//2
    
    def forward(self, signal_input: TorchSignal, task_info: torch.Tensor, signal_pilot: TorchSignal=TorchSignal()) -> Union[TorchSignal, list]:
        '''
        Input:
            signal_input: [batch, L*sps, Nmodes]
            task_info: [batch, 4]
            signal_pilot: [batch, L, Nmodes]
        Output:

        '''
        signal = self.ldbp(signal_input, task_info)
        signal = self.adf(signal, signal_pilot, task_info)
        return signal
    
    def change_mode(self, mode:str, batch_size:int=-1):
        '''
        change mode to 'train' or 'test'.
        '''
        self.mode = mode
        self.adf.change_mode(mode)
        if mode == 'test' and batch_size != -1:
            self.adf.change_batchsize(batch_size)

    def to(self, device):
        '''
        chance device of DSP.
        '''
        self.ldbp = self.ldbp.to(device)
        self.adf = self.adf.to(device)
        return self
    
    def show_lr(self, signal_input: TorchSignal, task_info: torch.Tensor, signal_pilot: TorchSignal=TorchSignal()):
        '''
            signal_input: [batch, L*sps, Nmodes]
            task_info: [batch, 4]
            signal_pilot: [batch, L, Nmodes]
        '''
        signal = self.ldbp(signal_input, task_info)
        lrs = self.adf(signal, signal_pilot, task_info, show_lr=True)
        return lrs


if __name__ == '__main__':
    import pickle, torch, numpy as np, time
    from torch.optim import Adam
    from .core import TorchInput, TorchSignal, TorchTime
    from .dataloader import get_data, signal_dataset

    train_data, info = get_data('data/train_data_few.pkl')
    batch_size = 360
    tbpl  = 200
    iters_pre_batch = 200
    dataset = signal_dataset(train_data, batch_size=batch_size, shuffle=True)

    # Define model
    DBP_info = {'step':5, 'dtaps': 5421,  'ntaps':401, 'type': 'MetaDBP', 'Nmodes':1,
            'L':2000e3, 'D':16.5, 'Fc':299792458/1550E-9, 'gamma':0.0016567,
            'task_dim':4, 'task_hidden_dim': 100}

    ADF_info = {'type':'metalstmplus' ,'mimotaps': 32, 'Nmodes':1, 
            'meta_args': {'step_max': 5e-2, 'hiddden_dim': 16, 'num_layers': 2}}


    net = DSP(DBP_info, ADF_info, batch_size=batch_size)
    optimizer = Adam(net.parameters(), lr=1e-4)
    device = 'cuda:0'
    L = tbpl + net.overlaps
    net = net.to(device)
    loss_list = []

    def MTLoss(predict, truth):
        return torch.sum(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=(-2,-1)))) 
    
    loss_fn = MTLoss



    for epoch in range(1):
        dataset = signal_dataset(train_data, batch_size=batch_size, shuffle=True)
        for b,data in enumerate(dataset):
            net.adf.init_state(batch_size=data.signal_input.val.shape[0])
            net = net.to(device)
            for i in range(20):
                t0 = time.time()
                x = data.get_data(L, i*tbpl).to(device)
                y = net(x.signal_input, x.task_info, x.signal_output)   # [B, L, N]
                truth = x.signal_output.val[:, y.t.start:y.t.stop]      # [B, L, N]
                loss = loss_fn(y.val, truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.adf.detach_state()
                t1 = time.time()
                if i % 1 == 0:
                    print(f'Epoch {epoch} data batch {b}/{dataset.batch_number()} iter {i}/100:  {loss.item()}     time cost per iteration: {t1 - t0}',)
                loss_list.append(loss.item())
        break


    
    

