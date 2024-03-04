import torch.nn as nn, torch, numpy as np, torch, matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from .core import TorchSignal, TorchTime
from .layers import MLP, ComplexLinear, complex_weight_composition, ComplexConv1d, Id, StepFunction



class NonlienarFeatures(nn.Module):
    '''
    Generate Nonlinear features from optical signal.
    Attributes:
        Nmodes: int. Number of modes.
        rho, L: size parameter.
        type: 'full', 'reduce-1', 'reduce-2'
    '''
    def __init__(self, Nmodes=1, rho=1.0, L=50, index_type='full'):
        super(NonlienarFeatures, self).__init__()
        assert Nmodes == 1 or Nmodes == 2, 'Nmodes should be 1 or 2.'
        self.Nmodes = Nmodes
        self.rho = rho 
        self.L = L
        self.index_type = index_type
        self.index = self.get_index()
        self.hdim = len(self.index)

    
    def valid_index(self, m,n):
        if self.index_type == 'full':
            return abs(m*n) <= self.rho * self.L //2
        elif self.index_type == 'reduce-1':
            return (abs(m*n) <= self.rho * self.L //2) and (n >= m)
        elif self.index_type == 'reduce-2':
            return (abs(m*n) <= self.rho * self.L //2) and (n > abs(m))
        
    def get_index(self):
        '''
            Get index set for pertubation.
        '''
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if self.valid_index(m,n):
                    S.append((m,n))
        return S 
    
    def get_mask(self):
        mask = []
        for m,n in self.index:
            if m*n == 0:
                mask.append(1)
            else:
                mask.append(0)
        return torch.tensor(mask, dtype=torch.float32)  # [hdim]
    
    def triplets(self, U, V, W, m,n):
        '''
        Generate triplets from U, V, W.
        U, V, W: [batch, L, Nmodes]  --> [batch, L, Nmodes]
        '''
        if self.Nmodes == 1:
            return torch.roll(U, m, dims=-2) * torch.roll(V, n, dims=-2) *torch.roll(W, m+n, dims=-2).conj()
        else:
            A = torch.roll(U, m, dims=-2) * torch.roll(W, m + n, dims=-2).conj() 
            return (A + A.roll(1, dims=-1)) * torch.roll(V, n, dims=-2)

    def forward(self, U, V, W):
        '''
        Generate Nonlinear features from optical signal.
        if am_split == False:
            U, V, W: [batch, L, Nmodes]  --> [batch, L, Nmodes, hdim]
        else:
            U, V, W: [batch, L, Nmodes]  --> [batch, L, Nmodes, hdim*2]
        '''
        Es = []
        if self.index_type == 'full':
            for m,n in self.index:
                Es.append(self.triplets(U, V, W, m,n))
        elif self.index_type == 'reduce-1':
            for m,n in self.index:
                if n == m:
                    Es.append(self.triplets(U, V, W, m,n))
                else:
                    Es.append(self.triplets(U, V, W, m,n) + self.triplets(U, V, W, n,m))
        elif self.index_type == 'reduce-2':
            for m,n in self.index:
                if n == m:
                    if m == 0:
                        Es.append(self.triplets(U, V, W, m,m))
                    else:
                        Es.append(self.triplets(U, V, W, m,m) + self.triplets(U, V, W, -m,-m))
                else:
                    if m+n == 0:
                        Es.append(self.triplets(U, V, W, m,n) + self.triplets(U, V, W, n,m))
                    else:
                        Es.append(self.triplets(U, V, W, m,n) + self.triplets(U, V, W, n,m) + self.triplets(U, V, W, -m, -n) + self.triplets(U, V, W, -n, -m))
        return torch.stack(Es, dim=-1) 
    

class FcPredict(nn.Module):
    '''
    Combine Nonlinear features from optical signal.
    Attributes:
        Nmodes: int. Number of modes.
    '''
    def __init__(self, hdim, Nmodes, pol_seperation=False):
        super(FcPredict, self).__init__()
        self.Nmodes = Nmodes
        self.hdim = hdim
        self.pol_seperation = pol_seperation

        if self.pol_seperation and self.Nmodes == 2:
            self.fc_x = ComplexLinear(hdim, 1, bias=False)
            self.fc_y = ComplexLinear(hdim, 1, bias=False)

        else:
            self.fc = ComplexLinear(hdim, 1)
    
    def forward(self, features):
        '''
        features: [batch, L, Nmodes, hdim]
        '''
        if self.pol_seperation and self.Nmodes == 2:
            return torch.cat([self.fc_x(features[:,:,0,:]), self.fc_y(features[:,:,0,:])], dim=-1)  # [batch, L, Nmodes]
        else:
            return self.fc(features)[..., 0]


class NNPredict(nn.Module):
    '''
    Combine Nonlinear features from optical signal.
    Attributes:
        Nmodes: int. Number of modes.
    '''
    def __init__(self, hdim, Nmodes, pol_seperation=False, scale=0.01, hidden_size=[2, 20], dropout=0.5, activation='leaky_relu', use_bias=True):
        super(NNPredict, self).__init__()
        self.Nmodes = Nmodes
        self.hdim = hdim
        self.pol_seperation = pol_seperation

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        self.activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            'id': Id(),
        }
        if self.pol_seperation and self.Nmodes == 2:
            self.nn_x = self.get_nn()
            self.nn_y = self.get_nn()

        else:
            self.nn = self.get_nn()

        self.scale = scale
    
    def get_nn(self):
        act = self.activations.get(self.activation, nn.ReLU())  # Default to ReLU if not found
        return nn.Sequential(
            nn.Linear(2*self.hdim, self.hidden_size[0], bias=self.use_bias),
            act,
            nn.Linear(self.hidden_size[0], self.hidden_size[1], bias=self.use_bias),
            act,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size[1], 2, bias=self.use_bias),
        )

    def c2r(self, x):
        '''
        x: [B, W, Nmodes, hdim]
        '''
        return torch.cat([x.real, x.imag], dim=-1)

    def r2c(self, x):
        '''
        x: [B, W, Nmodes, 2]
        '''
        return torch.complex(x[...,0], x[...,1]) 

    def forward(self, features):
        '''
        features: [batch, L, Nmodes, hdim]
        '''
        if self.pol_seperation and self.Nmodes == 2:
            return self.scale * torch.cat([self.r2c(self.nn_x(self.c2r(features[:,:,0,:]))), self.r2c(self.nn_y(self.c2r(features[:,:,0,:])))], dim=-1)  # [batch, L, Nmodes]
        else:
            return self.scale * self.r2c(self.nn(self.c2r(features)))


        
class FoPBC(nn.Module):
    '''
        First order PBC.
        Attributes:
            rho: float. Nonlinear parameter.
            L: int. Length of signal.
            Nmodes: int. Number of modes.
            index_type: str. 'full', 'reduce-1', 'reduce-2'
            pol_seperation: bool. If True, predict x and y seperately.
    '''
    def __init__(self, rho: float, L: int, Nmodes: int, index_type:str='reduce-2', pol_seperation:bool=False):
        super(FoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.overlaps = L
        self.Nmodes = Nmodes
        self.index_type = index_type
        self.pol_seperation = pol_seperation   
        self.nonlinear_features = NonlienarFeatures(Nmodes=Nmodes, rho=rho, L=L, index_type=index_type)
        self.predict = FcPredict(self.nonlinear_features.hdim, Nmodes, pol_seperation=pol_seperation)
    
    def get_power(self, task_info, device):
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/self.Nmodes   # [batch] or ()
        P = P.to(device)
        return P
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        E = signal.val             # [batch, W, Nmodes]
        P = self.get_power(task_info, E.device)   # [batch,]
        features = self.nonlinear_features(E, E, E)  # [batch, L, Nmodes, hdim]
        E = E + self.predict(features) * P[...,None, None]  # [batch, L, Nmodes]
        return TorchSignal(val=E[...,(self.overlaps//2):-(self.overlaps//2),:], t=TorchTime(signal.t.start + (self.overlaps//2), signal.t.stop - (self.overlaps//2), signal.t.sps))

    def get_C(self):
        if self.pol_seperation:
            return {'index': self.nonlinear_features.index, 
                    'Cx': self.predict.fc_x.real.weight.data.squeeze() + 1j*self.predict.fc_x.imag.weight.data.squeeze(), 
                    'Cy': self.predict.fc_y.real.weight.data.squeeze() + 1j*self.predict.fc_y.imag.weight.data.squeeze()}
        else:
            return {'index': self.nonlinear_features.index, 'C': self.predict.fc.real.weight.data.squeeze() + 1j*self.predict.fc.imag.weight.data.squeeze()} 

    def scatter_C(self, x, y, values, s=3, vmax=-1.5, vmin=-5, title='example'):
        if self.index_type == 'full':
            plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        elif self.index_type == 'reduce-1':
            plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
            plt.scatter(y, x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        elif self.index_type == 'reduce-2':
            plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
            plt.scatter(y, x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
            plt.scatter(-x, -y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
            plt.scatter(-y, -x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.colorbar(label='Value')
        plt.xlabel('m Coordinate')
        plt.ylabel('n Coordinate')
        plt.title(f'Heatmap of C_m,n (log10 scale) -- {title}')

    def show_C(self, figsize=(6,6), dpi=200, s=3, vmax=-1.5, vmin=-5):
        C = self.get_C()
        index = C['index']
        x,y = zip(*index)
        if self.pol_seperation:
            Cx = C['Cx']
            Cy = C['Cy']
            value_x = np.log10(np.abs(Cx))
            value_y = np.log10(np.abs(Cy))
            plt.figure(figsize=(figsize[0]*2, figsize[1]), dpi=dpi)
            plt.subplot(121)
            self.scatter_C(x, y, value_x, s, vmax=vmax, vmin=vmin, title='C_x')  # `cmap`指定颜色映射，`s`指定点的大小
            
            plt.subplot(122)
            self.scatter_C(x, y, value_y, s, vmax=vmax, vmin=vmin, title='C_y')
        else:
            plt.figure(figsize=figsize, dpi=dpi)
            C = C['C']
            value = np.log10(np.abs(C))
            self.scatter_C(x, y, value, s, vmax=vmax, vmin=vmin, title='C')



class ERPFoPBC(FoPBC):
    '''
        ERP  first order PBC.
        Attributes:
            rho: float. Nonlinear parameter.
            L: int. Length of signal.
            Nmodes: int. Number of modes.
            index_type: str. 'full', 'reduce-1', 'reduce-2'
            pol_seperation: bool. If True, predict x and y seperately.
    '''
    def __init__(self, rho: float, L: int, Nmodes: int, index_type:str='reduce-2', pol_seperation:bool=False):
        super(ERPFoPBC, self).__init__(rho, L, Nmodes, index_type, pol_seperation)
        self.C0_real = nn.Parameter(torch.zeros(Nmodes, dtype=torch.float), requires_grad=True)
        self.C0_imag = nn.Parameter(torch.zeros(Nmodes, dtype=torch.float), requires_grad=True)
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        E = signal.val             # [batch, W, Nmodes]
        P = self.get_power(task_info, E.device)   # [batch,]
        mask = self.nonlinear_features.get_mask().to(E.device)  # [hdim]
        C0 = self.C0_real + 1j*self.C0_imag

        features = self.nonlinear_features(E, E, E)  # [batch, L, Nmodes, hdim]
        E = (1 + C0)*E + self.predict(features) * P[...,None, None]  # [batch, L, Nmodes]
        return TorchSignal(val=E[...,(self.overlaps//2):-(self.overlaps//2),:], t=TorchTime(signal.t.start + (self.overlaps//2), signal.t.stop - (self.overlaps//2), signal.t.sps))

class NNFoPBC(nn.Module):
    '''
        First order PBC.
        Attributes:
            rho: float. Nonlinear parameter.
            L: int. Length of signal.
            Nmodes: int. Number of modes.
            index_type: str. 'full', 'reduce-1', 'reduce-2'
            pol_seperation: bool. If True, predict x and y seperately.
    '''
    def __init__(self, rho: float, L: int, Nmodes: int, index_type:str='reduce-2', pol_seperation:bool=False, **kwargs):
        super(NNFoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.overlaps = L
        self.Nmodes = Nmodes
        self.index_type = index_type
        self.pol_seperation = pol_seperation   
        self.nonlinear_features = NonlienarFeatures(Nmodes=Nmodes, rho=rho, L=L, index_type=index_type)
        self.predict = NNPredict(self.nonlinear_features.hdim, Nmodes, pol_seperation=pol_seperation, **kwargs)
    
    def get_power(self, task_info, device):
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/self.Nmodes   # [batch] or ()
        P = P.to(device)
        return P
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        E = signal.val             # [batch, W, Nmodes]
        P = self.get_power(task_info, E.device)   # [batch,]
        features = self.nonlinear_features(E, E, E)  # [batch, L, Nmodes, hdim]
        E = E + self.predict(features* torch.sqrt(P[...,None, None, None])**3) / torch.sqrt(P[...,None, None])   # [batch, L, Nmodes]
        return TorchSignal(val=E[...,(self.overlaps//2):-(self.overlaps//2),:], t=TorchTime(signal.t.start + (self.overlaps//2), signal.t.stop - (self.overlaps//2), signal.t.sps))



class AmFoPBC(FoPBC):
    '''
        Add-Multiply  first order PBC.
        Attributes:
            rho: float. Nonlinear parameter.
            L: int. Length of signal.
            Nmodes: int. Number of modes.
            index_type: str. 'full', 'reduce-1', 'reduce-2'
            pol_seperation: bool. If True, predict x and y seperately.
    '''
    def __init__(self, rho: float, L: int, Nmodes: int, index_type:str='reduce-2', pol_seperation:bool=False):
        super(AmFoPBC, self).__init__(rho, L, Nmodes, index_type, pol_seperation)
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        E = signal.val             # [batch, W, Nmodes]
        P = self.get_power(task_info, E.device)   # [batch,]
        mask = self.nonlinear_features.get_mask().to(E.device)  # [hdim]

        features = self.nonlinear_features(E, E, E)  # [batch, L, Nmodes, hdim]
        M = self.predict(features * mask) * P[...,None, None]  # [batch, L, Nmodes]
        A = self.predict(features * (1-mask)) * P[...,None, None]  # [batch, L, Nmodes]
        E = E * torch.exp(M / E) + A  # [batch, L, Nmodes]
        return TorchSignal(val=E[...,(self.overlaps//2):-(self.overlaps//2),:], t=TorchTime(signal.t.start + (self.overlaps//2), signal.t.stop - (self.overlaps//2), signal.t.sps))


class MySoPBC(nn.Module):

    def __init__(self, rho: float, L: int, Nmodes: int, index_type:str='reduce-2', pol_seperation:bool=False):
        super(MySoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.overlaps = L * 2
        self.Nmodes = Nmodes
        self.index_type = index_type
        self.pol_seperation = pol_seperation   
        self.nonlinear_features = NonlienarFeatures(Nmodes=Nmodes, rho=rho, L=L, index_type=index_type)
        self.predict1 = FcPredict(self.nonlinear_features.hdim, Nmodes, pol_seperation=pol_seperation)
        self.predict2 = FcPredict(self.nonlinear_features.hdim, Nmodes, pol_seperation=pol_seperation)
        self.predict3 = FcPredict(3*self.nonlinear_features.hdim, Nmodes, pol_seperation=pol_seperation)
    
    def get_power(self, task_info, device):
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/self.Nmodes   # [batch] or ()
        P = P.to(device)
        return P
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        E = signal.val             # [batch, W, Nmodes]
        P = self.get_power(task_info, E.device)   # [batch,]
        mask = self.nonlinear_features.get_mask().to(E.device)  # [hdim]
        features = self.nonlinear_features(E, E, E)  # [batch, L, Nmodes, hdim]
        M = self.predict1(features * mask) * P[...,None, None]  # [batch, L, Nmodes]
        A = self.predict1(features * (1-mask)) * P[...,None, None]  # [batch, L, Nmodes]
        
        E1 = self.predict2(features)  # [batch, W, Nmodes]
        F1 = self.nonlinear_features(E1, E, E)
        F2 = self.nonlinear_features(E, E1, E)
        F3 = self.nonlinear_features(E, E, E1)
        E2 = self.predict3(torch.cat([F1, F2, F3], dim=-1))  # [batch, W, Nmodes]

        E = E * torch.exp(M / E) + A  + E2*P[...,None, None]**2  # [batch, L, Nmodes]
        return TorchSignal(val=E[...,(self.overlaps//2):-(self.overlaps//2),:], t=TorchTime(signal.t.start + (self.overlaps//2), signal.t.stop - (self.overlaps//2), signal.t.sps))




class MultiStepPBC(nn.Module):
    def __init__(self, steps=2, fo_type='FoPBC', **kwargs):
        '''
        L propto Rs^2
        '''
        super(MultiStepPBC, self).__init__()
        self.steps = steps
        fo_models = {
            'FoPBC': FoPBC,
            'ERPFoPBC': ERPFoPBC,
            'AmFoPBC': AmFoPBC,
            'NNFoPBC': NNFoPBC,
        }  
        module = fo_models[fo_type]
        self.HPBC_steps = nn.ModuleList([module(**kwargs) for i in range(steps)])
        self.overlaps = sum([model.overlaps for model in self.HPBC_steps]) # type: ignore

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        for i in range(self.steps):
            signal = self.HPBC_steps[i](signal, task_info)
        return signal
    

models_new = {
            'FoPBC(new)': FoPBC,
            'ERPFoPBC(new)': ERPFoPBC,
            'NNFoPBC(new)': NNFoPBC,
            'AmFoPBC(new)': AmFoPBC,
            'MultiStepPBC(new)': MultiStepPBC,
            'MySoPBC(new)': MySoPBC,
        }  

if __name__ == '__main__':

    import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time
    from src.TorchDSP.dataloader import signal_dataset, get_k_batch
    from src.TorchDSP.baselines import CDCDSP
    from src.JaxSimulation.utils import show_symb
    from src.TorchSimulation.receiver import  BER 
    from src.TorchDSP.core import TorchSignal,TorchTime

    device = 'cpu'

    train_y, train_x,train_t = pickle.load(open('data/Nmodes2/train_afterCDCDSP.pkl', 'rb'))
    k = get_k_batch(1, 20, train_t)
    train_signal = TorchSignal(train_y[k], TorchTime(0,0,1)).to(device)
    train_z = train_t[k].to(device)

    print(train_y.shape)
    print(train_x.shape)
    
    signal = train_signal.get_slice(1000, 0)
    # net = FoPBC(1, 50, 2)
    # net = AmFoPBC(1, 50, 2)
    # net = MySoPBC(1, 50, 2)
    # net = ERPFoPBC(1, 50, 2)
    net = NNFoPBC(1, 50, 2)
    print(net)
    net = net.to(device)
    signal_out = net(signal, train_z)
    print(signal.val.shape)
    print(signal_out.val.shape)