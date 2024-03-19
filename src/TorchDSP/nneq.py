import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional
from .core import TorchSignal, TorchTime
from .layers import MLP, ComplexLinear

class PBC(nn.Module):
    '''
    The first order Pertubation based compensation.
    Attributes:
        rho, L: parameter for choose C index.
        Rs: symbol rate.
        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, Rs = torch.tensor([20, 40, 80, 160])):
        '''
        L propto Rs^2
        '''
        super(PBC, self).__init__()
        self.rho = rho
        self.L = L
        self.index = self.get_index()
        self.Rs = Rs # [Gbps]
        self.overlaps = self.L

        self.C_real = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C_imag = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  
    
    def get_C(self) -> np.ndarray:
        '''
            Get pertubation coefficients.
        '''
        C = torch.complex(self.C_real, self.C_imag)
        return C.detach().to('cpu').numpy()

    def get_index(self) -> list:
        '''
            Get pertubation indexes.
            S = {(m,n)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2}
        '''
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if abs(m*n) <= self.rho * self.L //2:
                    S.append((m,n))
        return S

    def choose_C(self, task_info):
        '''
        task_info: P,Fi,Fs,Nch 
        '''
        if task_info == None:
            return torch.complex(self.C_real, self.C_imag)[0]
        _, indices = self.Rs.sort()
        x = (task_info[..., 2] / 2e9 ).to(torch.int)
        ind = (x.unsqueeze(-1) == self.Rs[indices]).nonzero(as_tuple=True)[1]
        C = torch.complex(self.C_real, self.C_imag)
        return C[ind]


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
        C = self.choose_C(task_info)  # [batch, len(C)]  or [len(C)]
        E = signal.val                # [batch, L, Nmodes] or [L, Nmodes]
        t = signal.t                  # [batch, 4]  or [4]
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/E.shape[-1]   # [batch] or ()
        # P = torch.tensor(1) if task_info == None else 1e-3*10**(task_info[:,0]/10)/E.shape[-1] 
        # P = torch.tensor(1) if task_info == None else torch.ones(task_info.shape[0])   # [batch] or ()
        P = P.to(E.device)
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                Emn = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * C[...,i, None, None]
                Es.append(Emn)
        elif E.shape[-1] == 2:
            for i,(m,n) in enumerate(self.index):
                A = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2) * C[...,i, None, None]
                Es.append(Emn)
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')

        Eo = E + (torch.sqrt(P)**3)[...,None,None] * sum(Es)
        return TorchSignal(val=Eo[...,self.L:-self.L,:], t=TorchTime(t.start + self.L, t.stop - self.L, t.sps))
    

class SOPBC(nn.Module):
    '''
    the Seccond Order Pertubation based compensation layer.
    paper: Second-Order Perturbation Theory-Based Digital Predistortion for Fiber Nonlinearity Compensation
    Attributes:
        rho, L: parameter for choose C index.
        Rs: symbol rate.
        overlaps: int. 
        C_real, C_imag: torch.nn.Parameter. 
    '''
    def __init__(self, rho=1.0, L=50, Lk=10,  Rs = torch.tensor([20, 40, 80, 160])):
        '''
        L propto Rs^2
        '''
        super(SOPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.Lk = Lk
        self.index = self.get_index()
        self.Rs = Rs                   # [Gbps]
        self.overlaps = self.L

        self.pbc = PBC(rho, L, Rs)

        self.C_real_1 = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C_imag_1 = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  

        self.C_real_2 = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C_imag_2 = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  
    
    def get_C(self) -> tuple:
        '''
            Get pertubation coefficients.
        '''
        C1 = torch.complex(self.C_real_1, self.C_imag_1)
        C2 = torch.complex(self.C_real_2, self.C_imag_2)
        return C1.detach().to('cpu').numpy(), C2.detach().to('cpu').numpy()

    def get_index(self) -> list:
        '''
            Get pertubation indexes.
            S = {(m,n,k)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2, |k|<= Lk/2}
        '''
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if abs(m*n) <= self.rho * self.L //2:
                    for k in range(-self.Lk//2, self.Lk//2 + 1):
                        S.append((m,n,k))
        return S

    def choose_C(self, task_info):
        '''
        task_info: P,Fi,Fs,Nch 
        '''
        if task_info == None:
            return torch.complex(self.C_real_1, self.C_imag_1)[0], torch.complex(self.C_real_2, self.C_imag_2)[0]
        _, indices = self.Rs.sort()
        x = (task_info[..., 2] / 2e9 ).to(torch.int)
        ind = (x.unsqueeze(-1) == self.Rs[indices]).nonzero(as_tuple=True)[1]
        C1 = torch.complex(self.C_real_1, self.C_imag_1)
        C2 = torch.complex(self.C_real_2, self.C_imag_2)
        return C1[ind], C2[ind]


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
        C1,C2 = self.choose_C(task_info)  # [batch, len(C)]  or [len(C)]
        E = signal.val                # [batch, L, Nmodes] or [L, Nmodes]
        t = signal.t                  # [batch, 4]  or [4]
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/E.shape[-1]   # [batch] or ()
        P = P.to(E.device)
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n,k) in enumerate(self.index):
                Emnk1 = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * torch.roll(E, k, dims=-2) * torch.roll(E, k, dims=-2).conj()  * C1[...,i, None, None]
                Emnk2 = torch.roll(E, n, dims=-2).conj() * torch.roll(E, m + n, dims=-2) * torch.roll(E, m, dims=-2).conj() * torch.roll(E, k, dims=-2) * torch.roll(E, k, dims=-2)  * C2[...,i, None, None]
                Emnk = Emnk1 + Emnk2
                Es.append(Emnk)
        elif E.shape[-1] == 2:
            raise ValueError('Not implemented')
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')
        
        E2 = (torch.sqrt(P)**5)[...,None,None] * sum(Es)     # 2nd pertubation value
        Eo = self.pbc(signal, task_info).val + E2[...,self.L:-self.L,:]
        return TorchSignal(val=Eo, t=TorchTime(t.start + self.L, t.stop - self.L, t.sps))


class PBCNN(nn.Module):
    '''
    The first order Pertubation based compensation + NN.
    Attributes:
        rho, L: parameter for choose C index.
        Rs: symbol rate.
        overlaps: int. 
        hidden_size: list with length = 2. [2,10]
        dropout: float. 0.5
        Nmodes: int. 1
        activation: str. 'relu', 'sigmoid', 'tanh', 'leaky_relu'
    '''
    def __init__(self, rho=1.0, L=50, Rs = torch.tensor([20, 40, 80, 160]), hidden_size=[2, 10], dropout=0.5, Nmodes=1, activation='relu'):
        '''
        L propto Rs^2
        '''
        super(PBCNN, self).__init__()
        self.rho = rho
        self.L = L
        self.index = self.get_index()
        self.Rs = Rs # [Gbps]
        self.Nmodes = Nmodes
        self.overlaps = self.L
        
        self.dropout = dropout
        self.activation = activation

        # Activation functions mapping
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
            # Add more activation functions if needed
        }
        act = activations.get(activation, nn.ReLU())  # Default to ReLU if not found

        # 2layer MLP
        self.nn = nn.Sequential(
            nn.Linear(2*len(self.index), hidden_size[0]),
            act,
            nn.Linear(hidden_size[0], hidden_size[1]),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_size[1], 2*Nmodes),
        )

    def get_index(self) -> list:
        '''
            Get pertubation indexes.
            S = {(m,n)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2}
        '''
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if abs(m*n) <= self.rho * self.L //2:
                    S.append((m,n))
        return S


    def nonlinear_features(self, E):
        '''
        E: [batch, W, Nmodes] or [W, Nmodes]
        '''
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                Emn = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2)
                Es.append(Emn)
        elif E.shape[-1] == 2:
            raise ValueError('Not implemented')
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')
        F = torch.cat(Es, dim=-1)  # [batch, L, Nmodes*len(S)] or [L, Nmodes*len(S)]
        return torch.cat([F.real, F.imag], dim=-1)  # [batch, L, Nmodes*len(S)*2] or [L, Nmodes*len(S)*2]
    
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
        features = self.nonlinear_features(signal.val) # [batch, W, Nmodes*len(S)*2] or [W, Nmodes*len(S)*2]
        features = features[..., self.L:-self.L,:]     # [batch, W-2L, Nmodes*len(S)*2] or [W-2L, Nmodes*len(S)*2]
        E = self.nn(features*torch.sqrt(P[...,None,None])**3)         # [batch, W-2L, 2*Nmodes] or [W-2L, 2*Nmodes]
        E = E.reshape(E.shape[:-1] + (self.Nmodes, 2))     # [batch, W-2L, Nmodes, 2] or [W-2L, Nmodes,2]
        E = torch.complex(E[...,0], E[...,1])          # [batch, W-2L, Nmodes] or [W-2L, Nmodes]
        E = E + signal.val[...,self.L:-self.L,:]       # [batch, W-2L, Nmodes] or [W-2L, Nmodes]
        return TorchSignal(val=E, t=TorchTime(signal.t.start + self.L, signal.t.stop - self.L, signal.t.sps))
    




class HPBC_steptest(nn.Module):
    def __init__(self, rho=1.0, L=50, Rs = torch.tensor([20, 40, 80, 160])):
        '''
        L propto Rs^2
        '''
        super(HPBC_steptest, self).__init__()
        self.rho = rho
        self.L = L
        self.index = self.get_index()
        self.Rs = Rs # [Gbps]

        self.C1_real = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C1_imag = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  

        self.C2_real = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C2_imag = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  


    def get_C(self) -> tuple:
        C1 = torch.complex(self.C1_real, self.C1_imag)
        C2 = torch.complex(self.C2_real, self.C2_imag)
        return C1.detach().to('cpu').numpy(), C2.detach().to('cpu').numpy()

    def get_index(self):
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if abs(m*n) <= self.rho * self.L //2:
                    S.append((m,n))
        return S

    def choose_C(self, task_info):
        '''
        task_info: P,Fi,Fs,Nch 
        '''
        if task_info == None:
            return torch.complex(self.C1_real, self.C1_imag)[0]
        _, indices = self.Rs.sort()
        x = (task_info[..., 2] / 2e9 ).to(torch.int)
        ind = (x.unsqueeze(-1) == self.Rs[indices]).nonzero(as_tuple=True)[1]
        C1 = torch.complex(self.C1_real, self.C1_imag)
        C2 = torch.complex(self.C2_real, self.C2_imag)
        return C1[ind], C2[ind]


    def forward(self, signal: TorchSignal, signal_nlets: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} =  sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        C1, C2 = self.choose_C(task_info)  # [batch, len(C)]  or [len(C)]
        E = signal.val                # [batch, L, Nmodes] or [L, Nmodes]
        F = signal_nlets.val
        t = signal.t                  # [batch, 4]  or [4]
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/E.shape[-1]   # [batch] or ()
        P = P.to(E.device)
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                Emn = torch.roll(E, n, dims=-2) * torch.roll(F, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * C1[...,i, None, None] + torch.roll(F, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * C2[...,i, None, None]
                Es.append(Emn)
        elif E.shape[-1] == 2:
            for i,(m,n) in enumerate(self.index):
                A = torch.roll(E, n, dims=-2) * torch.roll(F, m + n, dims=-2).conj()*C1[...,i, None, None] + torch.roll(F, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj()*C2[...,i, None, None]
                Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2)
                Es.append(Emn)
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')

        Eo = signal_nlets.val + (torch.sqrt(P)**3)[...,None,None] * sum(Es)
        return TorchSignal(val=E[...,self.L:-self.L,:], t=TorchTime(t.start + self.L, t.stop - self.L, t.sps)), TorchSignal(val=Eo[...,self.L:-self.L,:], t=TorchTime(t.start + self.L, t.stop - self.L, t.sps))
    


class HPBC_step(nn.Module):

    def __init__(self, rho=1.0, L=50, Rs = torch.tensor([20, 40, 80, 160])):
        '''
        L propto Rs^2
        '''
        super(HPBC_step, self).__init__()
        self.rho = rho
        self.L = L
        self.index = self.get_index()
        self.Rs = Rs # [Gbps]

        self.C_real = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)] 
        self.C_imag = nn.Parameter(torch.zeros(len(self.Rs), len(self.index)), requires_grad=True)   # [num_C, len(C)]  


    def get_C(self):
        C = torch.complex(self.C_real, self.C_imag)
        return C.detach().to('cpu').numpy()

    def get_index(self):
        S = []
        for m in range(-self.L//2, self.L//2 + 1):
            for n in range(-self.L//2, self.L//2 + 1):
                if abs(m*n) <= self.rho * self.L //2:
                    S.append((m,n))
        return S

    def choose_C(self, task_info):
        '''
        task_info: P,Fi,Fs,Nch 
        '''
        if task_info == None:
            return torch.complex(self.C_real, self.C_imag)[0]
        _, indices = self.Rs.sort()
        x = (task_info[..., 2] / 2e9 ).to(torch.int)
        ind = (x.unsqueeze(-1) == self.Rs[indices]).nonzero(as_tuple=True)[1]
        C = torch.complex(self.C_real, self.C_imag)
        return C[ind]


    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        C = self.choose_C(task_info)  # [batch, len(C)]  or [len(C)]
        E = signal.val                # [batch, L, Nmodes] or [L, Nmodes]
        t = signal.t                  # [batch, 4]  or [4]
        P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/E.shape[-1]   # [batch] or ()
        P = P.to(E.device)
        Es = []
        if E.shape[-1] == 1:
            for i,(m,n) in enumerate(self.index):
                Emn = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() * torch.roll(E, m, dims=-2) * C[...,i, None, None]
                Es.append(Emn)
        elif E.shape[-1] == 2:
            for i,(m,n) in enumerate(self.index):
                A = torch.roll(E, n, dims=-2) * torch.roll(E, m + n, dims=-2).conj() 
                Emn = (A + A.roll(1, dims=-1)) * torch.roll(E, m, dims=-2) * C[...,i, None, None]
                Es.append(Emn)
        else:
            raise ValueError('H.shape[-1] should be 1 or 2')

        Eo = E + (torch.sqrt(P)**3)[...,None,None] * sum(Es)
        return TorchSignal(val=Eo[...,self.L:-self.L,:], t=TorchTime(t.start + self.L, t.stop - self.L, t.sps))
    

class HPBC(nn.Module):

    def __init__(self, rho=1.0, L=50, Rs = torch.tensor([20, 40, 80, 160]), steps=2):
        '''
        L propto Rs^2
        '''
        super(HPBC, self).__init__()
        self.rho = rho
        self.L = L
        self.Rs = Rs # [Gbps]
        self.steps = steps
        self.overlaps = self.L * steps
        self.HPBC_steps = nn.ModuleList([HPBC_step(rho, L, Rs) for i in range(steps)])

    def forward(self, signal: TorchSignal, task_info: Union[torch.Tensor,None] = None):
        '''
        E: [batch, L, Nmodes] or [L, Nmodes]
        task_info: P,Fi,Fs,Nch 
        O_{b,k,i} = sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i}
        '''
        for i in range(self.steps):
            signal = self.HPBC_steps[i](signal, task_info)
        return signal
    




class eqMLP(nn.Module):
    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int, Nmodes=2, widths=[149, 132, 596], res_net=True):
        super(eqMLP, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.flatten = nn.Flatten()
        self.widths = widths
        self.fc_layers = nn.ModuleList()

        self.fc_layers.append(nn.Linear(M*Nmodes*2, widths[0], bias=False))
        for i in range(len(widths)-1):
            self.fc_layers.append(nn.Linear(widths[i], widths[i+1], bias=False))
        self.fc_out = nn.Linear(widths[-1], Nmodes*2, bias=False)
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0)  # Adjust the mean and std as needed
        self.act = nn.Tanh()
    
    def forward(self, x:torch.Tensor):
        x0 = x[:, self.M //2,:]
        x = torch.cat([x.real, x.imag], dim=-1)  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]
        x = self.flatten(x)                       # float [B, M, Nmodes*2] -> float [B, M*Nmodes*2]

        for fc in self.fc_layers:
            x = self.act(fc(x))                   # float [B, *] -> float [B, widths[i]]
        x = self.fc_out(x)                 # float [B, widths[-1]] -> float [B, Nmodes*2]

        # convert to complex
        x = x.view(-1, self.Nmodes, 2)                 # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j)*x[..., 1]            # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net: x = x + x0
        return x


class eqCNNMLP(nn.Module):
    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int,  Nmodes:int=2, kernel_size=10, channels=470, widths=[456, 467], res_net=True):
        super(eqCNNMLP, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.conv1d = nn.Conv1d(in_channels=Nmodes*2, out_channels=channels, kernel_size=kernel_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc_layers = nn.ModuleList()

        self.dense1 = nn.Linear(channels * (M - kernel_size + 1), widths[0], bias=False)
        self.dense2 = nn.Linear(widths[0], widths[1], bias=False)
        self.dense3 = nn.Linear(widths[1], Nmodes*2, bias=False)
        nn.init.normal_(self.dense3.weight, mean=0.0, std=0)  # Adjust the mean and std as needed
        self.tanh = nn.Tanh()

    def forward(self, x):
        x0 = x[:, self.M //2,:]
        x = torch.cat([x.real, x.imag], dim=-1)   # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]
        
        x = x.permute(0, 2, 1)                     # float [B, M, Nmodes*2]  -> float [B, Nmodes*2, M]
        x = self.conv1d(x)                         # float [B, Nmodes*2, M]  -> float [B, channels, M-kernel_size+1]
        x = self.leaky_relu(x)                     # float [B, channels, M-kernel_size+1] -> float [B, channels, M-kernel_size+1]
        x = self.flatten(x)                        # float [B, channels, M-kernel_size+1] -> float [B, channels*(M-kernel_size+1)]
        x = self.tanh(self.dense1(x))              # float [B, channels*(M-kernel_size+1)] -> float [B, widths[0]]
        x = self.tanh(self.dense2(x))              # float [B, widths[0]] -> float [B, widths[1]]
        x = self.dense3(x)

        # convert to complex
        x = x.view(-1, self.Nmodes, 2)                 # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j)*x[..., 1]            # float [B, Nmodes, 2] -> complex [B, Nmodes]

        if self.res_net: x = x + x0
        return x

class eqBiLSTM(nn.Module):
    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int,  Nmodes=2, hidden_size=226, res_net=True):
        super(eqBiLSTM, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.lstm = nn.LSTM(input_size=Nmodes*2, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2*hidden_size * M, Nmodes*2, bias=False)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0)  # Adjust the mean and std as needed

    def forward(self, x):
        x0 = x[:, self.M //2,:]                  # Complex [B, M, Nmodes]  -> complex [B, Nmodes]
        x = torch.cat([x.real, x.imag], dim=-1)  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]

        x, _ = self.lstm(x)                       # float [B, M, Nmodes*2]  -> float [B, M, hidden_size*2]
        x = self.flatten(x)
        x = self.dense(x)                         # float [B, M*hidden_size*2] -> float [B, Nmodes*2]
        
        # convert to complex
        x = x.view(-1, self.Nmodes, 2)                 # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j)*x[..., 1]            # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net: x = x + x0
        return x

class eqCNNBiLSTM(nn.Module):
    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int, Nmodes=2, channels=244, kernel_size=10, hidden_size=113, res_net=True):
        super(eqCNNBiLSTM, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.conv1d = nn.Conv1d(in_channels=2*Nmodes, out_channels=channels, kernel_size=kernel_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2*hidden_size*(M-kernel_size+1), Nmodes*2, bias=False)
        nn.init.normal_(self.dense.weight, mean=0.0, std=0)  # Adjust the mean and std as needed

    def forward(self, x):
        x0 = x[:, self.M //2,:]
        x = torch.cat([x.real, x.imag], dim=-1)  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]

        x = x.permute(0, 2, 1)              # float [B, M, Nmodes*2]  -> float [B, Nmodes*2, M]
        x = self.conv1d(x)                  # float [B, Nmodes*2, M]  -> float [B, channels, M-kernel_size+1]
        x = self.leaky_relu(x)              # float [B, channels, M-kernel_size+1] -> float [B, channels, M-kernel_size+1]
        x = x.permute(0, 2, 1)              # float [B, channels, M-kernel_size+1] -> float [B, M-kernel_size+1, channels]
        x, _ = self.lstm(x)                 # float [B, M-kernel_size+1, channels] -> float [B, M-kernel_size+1, hidden_size*2]
        x = self.flatten(x)                 # float [B, M-kernel_size+1, hidden_size*2] -> float [B, M*hidden_size*2]
        x = self.dense(x)                   # float [B, M*hidden_size*2] -> float [B, Nmodes*2]
        
        # convert to complex
        x = x.view(-1, self.Nmodes, 2)           # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j)*x[..., 1]      # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net: x = x + x0
        return x
    

class eqID(nn.Module):

    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int, Nmodes=2):
        super(eqID, self).__init__()
        self.M = M

    def forward(self, x):

        return x[:,self.M//2,:]
    


class NNeq(nn.Module):
    def __init__(self, M, Nmodes, method='MLP', res_net=False):
        super(NNeq, self).__init__()
        self.M = M
        self.res_net = res_net

        self.Nmodes = Nmodes
        self.method = method
        if method == 'MLP':
            self.net = eqMLP(M, Nmodes, res_net=res_net)
        elif method == 'BiLSTM':
            self.net = eqBiLSTM(M, Nmodes, res_net=res_net)
        elif method == 'CNNBiLSTM':
            self.net = eqCNNBiLSTM(M, Nmodes, res_net=res_net)
        elif method == 'CNNMLP':
            self.net = eqCNNMLP(M, Nmodes, res_net=res_net)
        elif method == 'ID':
            self.net = eqID(M, Nmodes)
        else:
            raise ValueError('method should be MLP, BiLSTM, CNNBiLSTM or CNNMLP')
    
    def forward(self, x):
        # x: [batch, M, Nmodes]
        return self.net(x)
    

models = {
            'PBC':PBC,
            'HPBC':HPBC,
            'PBCNN':PBCNN,
            'SOPBC':SOPBC,
            'NNeq':NNeq
        }  

    
if __name__ == '__main__':

    import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time
    from TorchDSP.dataloader import signal_dataset, get_k_batch
    from TorchDSP.baselines import CDCDSP
    from JaxSimulation.utils import show_symb
    from TorchSimulation.receiver import  BER 
    from TorchDSP.core import TorchSignal,TorchTime

    device = 'cpu'

    train_y, train_x,train_t = pickle.load(open('data/train_data_afterCDCDSP.pkl', 'rb'))
    k = get_k_batch(1, 20, train_t)
    train_signal = TorchSignal(train_y[k], TorchTime(0,0,1)).to(device)
    train_z = train_t[k]
    
    signal = train_signal.get_slice(1000, 0)
    # net = SOPBC(rho=1.0, L=50, Lk=10, Rs = torch.tensor([20]))
    net = PBCNN(rho=1.0, L=50, Rs = torch.tensor([20]))
    # net = HPBC(rho=1.0, L=50, Rs = torch.tensor([20]))
    signal_out = net(signal, train_z)

    print(signal_out.val.shape)



