import torch.nn as nn, torch, numpy as np, torch, matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from .core import TorchSignal, TorchTime
from .layers import MLP, ComplexLinear, complex_weight_composition, ComplexConv1d


class BasePBC(nn.Module):
    '''
    Base Pertubation based compensation.
    Attributes:
        rho, L: parameter for choose C index.
        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(BasePBC, self).__init__()
        self.rho = rho
        self.L = L
        self.index_type = index_type
        self.index = self.get_index()
        self.overlaps = self.L

    def get_index(self) -> list:
        '''
            Get pertubation indexes.
            index_type == 'A':
                S_{A} = {(m,n)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2}   
                rho, L control size of S_{A}. 
                |S_{A}| = 2L log(L/2) + 2L
            index_type == 'B':
                S_{B} =  {(m,n)| |m| + |n|<= L/2}
                L control the size of S_{B}
                |S_{B}| = 2L^2
        '''
        S = []

        if self.index_type == 'A':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if abs(m*n) <= self.rho * self.L //2:
                        S.append((m,n))
        elif self.index_type == 'B':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if abs(m)+abs(n) <= self.L //2:
                        S.append((m,n))
        else:
            raise ValueError
        return S
    
    def get_C(self) -> np.ndarray:
        '''
            Get pertubation coefficients.  [1, len(S)]
            Each sub-class need to redefine.
        '''

        return np.zeros([1,len(self.index)], dtype=np.complex64)
    

    def show_coeff(self, title, vmin=-5, vmax=-1.5):
        C = self.get_C()
        Y = np.zeros_like(C)
        X = {}
        for i,(m,n) in enumerate(self.index):
            X[(m,n)] = C[0,i]
        for i,(m,n) in enumerate(self.index):
            Y[0,i] = (X[(m,n)]+X[(n,m)])/2

        values = np.log10(np.abs(Y))[0]   # type: ignore
        x, y = zip(*self.index)
        sc1 = plt.scatter(x, y, c=values, cmap='viridis', s=3, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        cbar1 = plt.colorbar(sc1, label='Value')  # 添加颜色条到第一个子图
        plt.xlabel('m Coordinate')
        plt.ylabel('n Coordinate')
        plt.title(f'Heatmap of C_m,n (log10 scale)--- {title}')
        plt.grid(True)
        return X

    
    def nonlinear_features(self, E):
        '''
        1 order PBC nonlinear features.
            E: [batch, W, Nmodes] or [W, Nmodes] -> [batch, W, Nmodes, len(S)] or [W, Nmodes,len(S)]
        '''
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                Emn = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2)
                Es.append(Emn)
        elif E.shape[-1] == 2:
            for i,(m,n) in enumerate(self.index):
                A = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2)
                Es.append(Emn)
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')
        F = torch.stack(Es, dim=-1)  # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        return F # [batch, W, Nmodes, len(S)] or [W, Nmodes,len(S)]



class FoPBC(BasePBC):
    '''
    The First Order Pertubation based compensation.
    Attributes:
        rho, L: parameter for choose C index.
        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(FoPBC, self).__init__(rho, L, index_type)
        self.nn = ComplexLinear(len(self.index), 1, bias=False)
        nn.init.zeros_(self.nn.real.weight)
        nn.init.zeros_(self.nn.imag.weight)

    def get_C(self):
        '''
            Get pertubation coefficients.
        '''
        C = self.nn.real.weight.data + (1j) * self.nn.imag.weight.data
        return C.detach().to('cpu').numpy()

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        '''
        Input:
            signal: val shape = [batch, L, Nmodes] or [L, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        '''
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/signal.val.shape[-1]   # [batch] or ()
        P = P.to(signal.val.device)
        features = self.nonlinear_features(signal.val)                       # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        features = features[..., (self.L//2):-(self.L//2),:,:]               # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]
        E = self.nn(features*torch.sqrt(P[...,None,None,None])**3)           # [batch, W-L, Nmodes, 1] or [W-L, Nmodes, 1]
        E = E[...,0]                                                         # [batch, W-L, Nmodes] or [W-L, Nmodes]
        E = E + signal.val[...,(self.L//2):-(self.L//2),:]                   # [batch, W-L, Nmodes] or [W-L, Nmodes]
        return TorchSignal(val=E, t=TorchTime(signal.t.start + (self.L//2), signal.t.stop - (self.L//2), signal.t.sps))
    

class FoPBCNN(BasePBC):

    def __init__(self, rho=1.0, L=50, index_type='A', hidden_size=[2, 10], dropout=0.5, activation='leaky_relu'):
        '''
        L propto Rs^2
        '''
        super(FoPBCNN, self).__init__(rho, L, index_type)

        self.dropout = dropout
        self.activation = activation
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            # Add more activation functions if needed
        }
        act = activations.get(activation, nn.ReLU())  # Default to ReLU if not found

        # 2layer MLP: R^{2*len(C)}  -> R^{2}
        self.nn = nn.Sequential(
            nn.Linear(2*len(self.index), hidden_size[0]),
            act,
            nn.Linear(hidden_size[0], hidden_size[1]),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size[1], 2),
        )

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        '''
        Input:
            signal: val shape = [batch, W, Nmodes] or [W, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        '''
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/signal.val.shape[-1]   # [batch] or ()
        P = P.to(signal.val.device)

        # truncated nonlinear features
        features = self.nonlinear_features(signal.val)             # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        features = features[..., (self.L//2):-(self.L//2),:,:]     # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]

        # complex -> real
        features = torch.cat([features.real, features.imag], dim=-1) # [batch, W-L, Nmodes, 2*len(S)] or [W-L, Nmodes, 2*len(S)]
        # Lienar layer
        E = self.nn(features*torch.sqrt(P[...,None,None,None])**3)   #  [batch, W-L, Nmodes, 2] or [W-L, Nmodes, 2]
        # real -> complex
        E = torch.complex(E[...,0], E[...,1])                        # [batch, W-L, Nmodes] or [W-L, Nmodes]

        # residual connection
        E = E + signal.val[...,(self.L//2):-(self.L//2),:]           # [batch, W-L, Nmodes] or [W-L, Nmodes]
        return TorchSignal(val=E, t=TorchTime(signal.t.start + (self.L//2), signal.t.stop - (self.L//2), signal.t.sps))
    
    

class HoPBC(nn.Module):
    '''
        High order PBC = Multi-step FOPBC.
    '''

    def __init__(self, rho=1.0, L=50, steps=2, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(HoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.index_type = index_type
        self.steps = steps
        self.overlaps = self.L * steps
        self.HPBC_steps = nn.ModuleList([FoPBC(rho, L, index_type) for i in range(steps)])

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        for i in range(self.steps):
            signal = self.HPBC_steps[i](signal, task_info)
        return signal


class SoPBC(nn.Module):
    '''
    the Seccond Order Pertubation based compensation layer.
    paper: Second-Order Perturbation Theory-Based Digital Predistortion for Fiber Nonlinearity Compensation
    Attributes:
        rho, L: parameter for choose C index. (SO part)   L <= L1
        rho1, L1: parameter for choose C index. (FO part),  default = rho, L 

        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, Lk=10, rho1=None, L1=None, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(SoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.rho1 = rho if rho1 == None else rho1
        self.L1 = L if L1 == None else L1
        self.Lk = Lk
        assert self.L <= self.L1, 'L should be less than L1'

        self.index_type = index_type
        self.index = self.get_index()
        self.overlaps = self.L1
        self.pbc = FoPBC(self.rho1, self.L1)
        self.nn1 = ComplexLinear(len(self.index), 1, bias=False)
        self.nn2 = ComplexLinear(len(self.index), 1, bias=False)
        nn.init.zeros_(self.nn1.real.weight)
        nn.init.zeros_(self.nn1.imag.weight)
        nn.init.zeros_(self.nn2.real.weight)
        nn.init.zeros_(self.nn2.imag.weight)

        
    def get_C(self) -> tuple:
        '''
            Get pertubation coefficients.
        '''
        C1 = self.nn1.real.weight.data + (1j) * self.nn1.imag.weight.data
        C2 = self.nn2.real.weight.data + (1j) * self.nn2.imag.weight.data
        return C1.detach().to('cpu').numpy(), C2.detach().to('cpu').numpy()

    def get_index(self) -> list:
        '''
            Get pertubation indexes.
            index_type == 'A':
                S_{A} = {(m,n,k)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2, |k|<= Lk/2}
                |S_{A}| = 2rhoL(log(rho L/2) + 1)*Lk
            index_type == 'B':
                S_{B} = {(m,n,k)| |m|+|n|<= L/2, |k|<= Lk/2}   
                |S_B| = 1/2*L^2L_k
            # !TODO
            2 order PBC How to choose index set ? 
        '''
        S = []

        if self.index_type == 'A':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if abs(m*n) <= self.rho * self.L //2:
                        for k in range(-self.Lk//2, self.Lk//2 + 1):
                            S.append((m,n,k))
        elif self.index_type == 'B':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if abs(m*n) <= self.rho * self.L //2:
                        for k in range(-self.Lk//2, self.Lk//2 + 1):
                            S.append((m,n,k))
        else:
            raise ValueError
        return S
    
    def nonlinear_features(self, E):
        '''
        2 order PBC nonlinear features.
            E: Complex [batch, W, Nmodes] or [W, Nmodes] -> Complex [batch, W, Nmodes, len(S)] or [W, Nmodes,len(S)]
        '''
        Es1 = []
        Es2 = []
        if E.shape[-1] == 1:
            for i,(m,n,k) in enumerate(self.index):
                Emnk1 = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * torch.roll(E, k, dims=-2) * torch.roll(E, k, dims=-2).conj()
                Emnk2 = torch.roll(E, n, dims=-2).conj() * torch.roll(E, m + n, dims=-2) * torch.roll(E, m, dims=-2).conj() * torch.roll(E, k, dims=-2) * torch.roll(E, k, dims=-2)
                Es1.append(Emnk1)
                Es2.append(Emnk2)
        elif E.shape[-1] == 2:
            raise ValueError('Not implemented')
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')
        F1 = torch.stack(Es1, dim=-1)  # [batch, L, Nmodes*len(S)] or [L, Nmodes*len(S)]
        F2 = torch.stack(Es2, dim=-1)  # [batch, L, Nmodes*len(S)] or [L, Nmodes*len(S)]
        return F1, F2                  # complex [batch, L, Nmodes, len(S)] or [L, Nmodes, len(S)]

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        '''
        Input:
            signal: val shape = [batch, L, Nmodes] or [L, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,p,i} = gamma^2 P0^{5/2} sum_{m,n,k} (E_{b, p+m, i} E_{b, p+m+n, i}^* E_{b, p+n, i} E_{b, p+k, i} E_{b, p+k, i}^* C_{1,m,n,k} +  E_{b, p+m, i}^* E_{b, p+m+n, i} E_{b, p+n, i}^* E_{b, p+k, i} E_{b, p+k, i} C_{2,m,n,k}) 
            Nmodes = 2:
                O_{b,p,i} = (8/9)^2 gamma^2 P0^{5/2} sum_{m,n,k} (E_{b,p+m,i}E_{b,p+m+n,i}^* + E_{b,p+m,-i}E_{b,p+m+n,-i}^*)E_{b,p+n,i}(E_{b,p+k,i}E_{b,p+k,i}^* + E_{b,p+k,-i}E_{b,p+k,-i}^*)C_{1,m,n,k}
                + (E_{b,p+m,i}^* E_{b,p+m+n,i} + E_{b,p+m,-i}^* E_{b,p+m+n,-i})E_{b,p+n,i}^*(E_{b,p+k,i}E_{b,p+k,i} + E_{b,p+k,-i}E_{b,p+k,-i})C_{2,m,n,k}
        '''
        E = signal.val                # [batch, W, Nmodes] or [W, Nmodes]
        t = signal.t                  # [batch, 4]  or [4]
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/E.shape[-1]   # [batch] or ()
        P = P.to(E.device)
        F1,F2 = self.nonlinear_features(E)                      # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        F1 = F1[..., (self.overlaps//2):-(self.overlaps//2),:,:]              # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]
        F2 = F2[..., (self.overlaps//2):-(self.overlaps//2),:,:]              # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]
        E1 = self.nn1(F1*torch.sqrt(P[...,None,None,None])**5)  # [batch, W-L, Nmodes, 1] or [W-L,Nmodes, 1]
        E2 = self.nn2(F2*torch.sqrt(P[...,None,None,None])**5)  # [batch, W-L, Nmodes, 1] or [W-L,Nmodes, 1]
        Eo = self.pbc(signal, task_info).val + E1[...,0] + E2[...,0]  # [batch, W-L, Nmodes] or [W-L,Nmodes]
        return TorchSignal(val=Eo, t=TorchTime(t.start + (self.overlaps//2), t.stop - (self.overlaps//2), t.sps))




class SymPBC(nn.Module):
    '''
    Symetric Pertubation based compensation.
    Attributes:
        rho, L: parameter for choose C index.
        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(SymPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.index_type = index_type
        self.index = self.get_index()
        self.overlaps = self.L

    def get_index(self):
        '''
            Get symetric pertubation indexes.
            S = {(m,n)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2, n>=|m|}
        '''
        S = []
        if self.index_type == 'A':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if (abs(m*n) <= self.rho * self.L //2) and (n >= abs(m)):
                        S.append((m,n))
        elif self.index_type == 'B':
            for m in range(-self.L//2, self.L//2 + 1):
                for n in range(-self.L//2, self.L//2 + 1):
                    if (abs(m) + abs(n) <=  self.L //2) and (n >= abs(m)):
                        S.append((m,n))
        else:
            raise ValueError
        return S
    
    def get_C(self) -> np.ndarray:
        '''
            Get pertubation coefficients.  [1, len(S)]
            Each sub-class need to redefine.
        '''

        return np.zeros([1,len(self.index)], dtype=np.complex64)
    

    def show_coeff(self, title='step 1', vmin=-5, vmax=-1.5):
        C = self.get_C()
        values = np.log10(np.abs(C))[0]   # type: ignore
        x, y = zip(*self.index)
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x, y, c=values, cmap='viridis', s=3, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(y, x, c=values, cmap='viridis', s=3, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-x, -y, c=values, cmap='viridis', s=3, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-y, -x, c=values, cmap='viridis', s=3, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.colorbar(label='Value')  # 添加颜色条到第一个子图
        plt.xlabel('m Coordinate')
        plt.ylabel('n Coordinate')
        plt.title(f'Heatmap of C_m,n (log10 scale)--- {title}')
        plt.grid(True)
        return C

    def nonlinear_features(self, E):
        '''
        1 order PBC nonlinear features.
            E: [batch, W, Nmodes] or [W, Nmodes] -> [batch, W, Nmodes, len(S)] or [W, Nmodes,len(S)]
        '''
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                if n > abs(m):
                    Emn = 2*(torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) + torch.roll(E, -n, dims=-2) * torch.roll(E, - m - n, dims=-2).conj() * torch.roll(E, -m, dims=-2))
                elif (m==0 and n==0):
                    Emn = E*E.conj()*E 
                else:
                    Emn = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) + torch.roll(E, -n, dims=-2) * torch.roll(E, - m - n, dims=-2).conj() * torch.roll(E, -m, dims=-2)
                Es.append(Emn)
        elif E.shape[-1] == 2:
            for i,(m,n) in enumerate(self.index):
                if n > abs(m):
                    A = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                    Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2)
                    A = torch.roll(E, m, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                    Enm = (A + A.roll(1, dims=-1)) * torch.roll(E, n, dims=-2)
                    A = torch.roll(E, -n, dims=-2) * torch.roll(E, -m - n, dims=-2).conj() 
                    Emn_ = (A + A.roll(1, dims=-1)) * torch.roll(E, -m, dims=-2)
                    A = torch.roll(E, -m, dims=-2) * torch.roll(E, -m - n, dims=-2).conj() 
                    Enm_ = (A + A.roll(1, dims=-1)) * torch.roll(E, -n, dims=-2)
                    Es.append(Emn+Enm+Emn_+Enm_)
                elif (m==0 and n==0):
                    A = E*E.conj()
                    Emn = (A + A.roll(1, dims=-1)) * E
                    Es.append(Emn)
                else:
                    A = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                    Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2)
                    A = torch.roll(E, -n, dims=-2) * torch.roll(E, -m - n, dims=-2).conj() 
                    Emn_ = (A + A.roll(1, dims=-1)) * torch.roll(E, -m, dims=-2)
                    Es.append(Emn+Emn_)

        else:
            raise ValueError('H.shape[-1] should be 1 or 2')
        F = torch.stack(Es, dim=-1)  # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        return F # [batch, W, Nmodes, len(S)] or [W, Nmodes,len(S)]



class SymFoPBC(SymPBC):

    def __init__(self, rho=1.0, L=50, index_type='A'):
        super(SymFoPBC, self).__init__(rho, L, index_type)
        self.nn = ComplexLinear(len(self.index), 1, bias=False)  # no bias term
        nn.init.zeros_(self.nn.real.weight)
        nn.init.zeros_(self.nn.imag.weight)

    def get_C(self):
        '''
            Get pertubation coefficients.
        '''
        C = self.nn.real.weight.data + (1j) * self.nn.imag.weight.data
        return C.detach().to('cpu').numpy()
    
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        '''
        Input:
            signal: val shape = [batch, L, Nmodes] or [L, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        '''
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/signal.val.shape[-1]   # [batch] or ()
        P = P.to(signal.val.device)
        features = self.nonlinear_features(signal.val)                       # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        features = features[..., (self.L//2):-(self.L//2),:,:]               # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]
        E = self.nn(features*torch.sqrt(P[...,None,None,None])**3)           # [batch, W-L, Nmodes, 1] or [W-L, Nmodes, 1]
        E = E[...,0]                                                         # [batch, W-L, Nmodes] or [W-L, Nmodes]
        E = E + signal.val[...,(self.L//2):-(self.L//2),:]                   # [batch, W-L, Nmodes] or [W-L, Nmodes]
        return TorchSignal(val=E, t=TorchTime(signal.t.start + (self.L//2), signal.t.stop - (self.L//2), signal.t.sps))


class SymFoPBCNN(SymPBC):

    def __init__(self, rho=1.0, L=50, index_type='A', hidden_size=[2, 10], dropout=0.5, activation='leaky_relu', use_bias=True, init_type=0):
        '''
        L propto Rs^2
        '''
        super(SymFoPBCNN, self).__init__(rho, L, index_type)

        self.dropout = dropout
        self.activation = activation
        self.use_bias = use_bias
        # Activation functions mapping
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            # Add more activation functions if needed
        }
        act = activations.get(activation, nn.ReLU())  # Default to ReLU if not found

        # 2layer MLP: R^{2*len(C)}  -> R^{2}
        self.nn = nn.Sequential(
            nn.Linear(2*len(self.index), hidden_size[0], bias=use_bias),
            act,
            nn.Linear(hidden_size[0], hidden_size[1], bias=use_bias),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size[1], 2, bias=use_bias),
        )

        if init_type == 0:
            nn.init.constant_(self.nn[0].weight, 0.0) # type: ignore
            nn.init.constant_(self.nn[2].weight, 0.0) # type: ignore
        elif init_type == 1:
            nn.init.constant_(self.nn[5].weight, 0.0) # type: ignore
        elif init_type == 2:
            nn.init.constant_(self.nn[0].weight, 0.0) # type: ignore
        elif init_type == 3:
            pass
        elif init_type == 4:
            dic = torch.load('/home/xiaoxinyu/TorchFiber/_models/SymFoPBC_L200_rho1_index_typeA_Nch1_Rs40_opt(Adam_lr1e-5)_loss(Mean)_Pch-1.ckpt200', map_location='cpu')
            weight = complex_weight_composition(dic['model']['nn.real.weight'].data, dic['model']['nn.imag.weight'].data)
            self.nn[0].weight.data = weight
            if use_bias: nn.init.zeros_(self.nn[0].bias)         # type: ignore
        else:
            raise ValueError('init_type should be 0, 1, 2, 3')

	

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:

        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/signal.val.shape[-1]   # [batch] or ()
        P = P.to(signal.val.device)

        # truncated nonlinear features
        features = self.nonlinear_features(signal.val)             # [batch, W, Nmodes, len(S)] or [W, Nmodes, len(S)]
        features = features[..., (self.L//2):-(self.L//2),:,:]     # [batch, W-L, Nmodes, len(S)] or [W-L, Nmodes, len(S)]

        # complex -> real
        features = torch.cat([features.real, features.imag], dim=-1) # [batch, W-L, Nmodes, 2*len(S)] or [W-L, Nmodes, 2*len(S)]
        # Lienar layer
        E = self.nn(features*torch.sqrt(P[...,None,None,None])**3)   #  [batch, W-L, Nmodes, 2] or [W-L, Nmodes, 2]
        # real -> complex
        E = torch.complex(E[...,0], E[...,1])                        # [batch, W-L, Nmodes] or [W-L, Nmodes]

        # residual connection
        E = E + signal.val[...,(self.L//2):-(self.L//2),:]           # [batch, W-L, Nmodes] or [W-L, Nmodes]
        return TorchSignal(val=E, t=TorchTime(signal.t.start + (self.L//2), signal.t.stop - (self.L//2), signal.t.sps))


class SymHoPBC(nn.Module):
    '''
        High order PBC = Multi-step FOPBC.
    '''

    def __init__(self, rho=1.0, L=50, steps=2, index_type='A'):
        '''
        L propto Rs^2
        '''
        super(SymHoPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.steps = steps
        self.index_type = index_type
        self.overlaps = self.L * steps
        self.HPBC_steps = nn.ModuleList([SymFoPBC(rho, L) for i in range(steps)])

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        for i in range(self.steps):
            signal = self.HPBC_steps[i](signal, task_info)
        return signal


class ConvPBC(nn.Module):

    def __init__(self, Nmodes=1, xpm_size=101, fwm_heads=16):
        super().__init__()
        self.Nmodes = Nmodes
        assert xpm_size % 2 == 1
        self.xpm_size = xpm_size
        self.fwm_size = xpm_size
        self.fwm_heads = fwm_heads
        self.overlaps = self.xpm_size - 1
        self.xpm_conv = nn.Conv1d(self.Nmodes, self.Nmodes, self.xpm_size)      # real convolution
        self.fwm_conv_m = ComplexConv1d(1, self.fwm_heads, self.fwm_size)   # complex convolution
        self.fwm_conv_n = ComplexConv1d(1, self.fwm_heads, self.fwm_size)   # complex convolution
        self.fwm_conv_k = ComplexConv1d(1, self.fwm_heads, self.fwm_size)   # complex convolution
        
    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None) -> TorchSignal:
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/signal.val.shape[-1]   # [batch] or ()
        P = P.to(signal.val.device)
        x = signal.val.transpose(1,2)  # x [B, M, L]
        phi = self.xpm_conv(torch.abs(x)**2)      # [B, M, L - xpm_size + 1]
        x_ = x.view(-1, x.shape[-1]).unsqueeze(1) # [B*M, 1, L]
        Am = self.fwm_conv_m(x_).view(x.shape[0], x.shape[1], self.fwm_heads, -1)       # [B, M, heads, L - fwm_size + 1]
        An = self.fwm_conv_n(x_).view(x.shape[0], x.shape[1], self.fwm_heads, -1)       # [B, M, heads, L - fwm_size + 1]
        Ak = self.fwm_conv_k(x_).view(x.shape[0], x.shape[1], self.fwm_heads, -1)       # [B, M, heads, L - fwm_size + 1]
        S = torch.sum(Am*Ak.conj(), dim=1)                                              # [B, heads, L - fwm_size + 1]
        E = x[:,:, self.xpm_size//2:-(self.xpm_size//2)]*torch.exp(1j*phi) + torch.sum(An*S.unsqueeze(1), dim=2)  # [B, M, L - xpm_size + 1]
        return  TorchSignal(val=E.transpose(1,2), t=TorchTime(signal.t.start + (self.xpm_size//2), signal.t.stop - (self.xpm_size//2), signal.t.sps))


class HoConvPBC(nn.Module):

    def __init__(self, steps=2, Nmodes=1, xpm_size=101, fwm_heads=16):
        '''
        L propto Rs^2
        '''
        super(HoConvPBC, self).__init__()
        self.Nmodes = Nmodes 
        self.xpm_size = xpm_size 
        self.fwm_heads = fwm_heads
        self.steps = steps
        self.overlaps = (self.xpm_size - 1) * steps
        self.ConvPBC_steps = nn.ModuleList([ConvPBC(Nmodes, xpm_size, fwm_heads) for i in range(steps)])

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        for i in range(self.steps):
            signal = self.ConvPBC_steps[i](signal, task_info)
        return signal


models = {
            'FoPBC': FoPBC,
            'SoPBC': SoPBC,
            'HoPBC': HoPBC,
            'FoPBCNN': FoPBCNN,
            'SymFoPBC': SymFoPBC,
            'SymFoPBCNN': SymFoPBCNN,
            'SymHoPBC': SymHoPBC
        }  



    
if __name__ == '__main__':

    import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time
    from src.TorchDSP.dataloader import signal_dataset, get_k_batch
    from src.TorchDSP.baselines import CDCDSP
    from src.JaxSimulation.utils import show_symb
    from src.TorchSimulation.receiver import  BER 
    from src.TorchDSP.core import TorchSignal,TorchTime

    device = 'cpu'

    train_y, train_x,train_t = pickle.load(open('data/train_data_afterCDCDSP.pkl', 'rb'))
    k = get_k_batch(1, 20, train_t)
    train_signal = TorchSignal(train_y[k], TorchTime(0,0,1)).to(device)
    train_z = train_t[k].to(device)

    print(train_y.shape)
    print(train_x.shape)
    
    signal = train_signal.get_slice(1000, 0)
    #net = FoPBC(rho=1.0, L=50)
    #net = SoPBC(rho=1.0, L=50, Lk=10)
    #net = HoPBC(rho=1.0, L=50, steps=2)
    #net = FoPBCNN(rho=1.0, L=50, hidden_size=[2, 10], dropout=0.5, activation='relu')
    #net = SymFoPBC(rho=1.0, L=50)
    #net = SymFoPBCNN(rho=1.0, L=50, hidden_size=[2, 10], dropout=0.5, activation='relu')
    #net = SymHoPBC(rho=1.0, L=50, steps=2)
    # net = ConvPBC()
    net = HoConvPBC()
    net = net.to(device)
    signal_out = net(signal, train_z)

    print(signal_out.val.shape)

