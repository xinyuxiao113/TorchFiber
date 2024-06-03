'''
    NN equalizer.
'''
import torch.nn as nn, torch, numpy as np, torch
from typing import Union, List, Tuple, Optional
from .core import TorchSignal, TorchTime
from .layers import MLP, ComplexLinear, ComplexConv1d
from src.TorchDSP.loss import MSE, Qsq
from src.TorchSimulation.receiver import BER


def T(U, V, W, m, n):
    '''
    nonlinear triplets.
    Input:
        U, V, W: [batch, M, Nmodes] 
        m, n: int
    Output:
        [batch, Nmodes]
    '''
    assert U.shape[1] % 2 == 1, 'M must be odd.'
    assert U.shape[-1] == 1 or U.shape[-1] == 2, 'Nmodes must be 1 or 2'

    p = U.shape[1] // 2
    if U.shape[-1] == 1:
        return U[:,p+m] * V[:,p+n] * W[:,p+m+n].conj()
    else:
        A = U[:,p+m] * W[:,p+m+n].conj() 
        return (A + A.roll(1, dims=-1)) * V[:,p+n]

def triplets(E, m, n):
    return T(E,E,E,m,n)

def get_power(task_info, Nmodes, device):
    '''
    task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]  
    Nmodes: int
    device: torch.device
    -> torch.Tensor [batch] 
    '''
    P = torch.tensor(1) if task_info == None else 10**(task_info[:,0]/10)/Nmodes   # [batch] or ()
    P = P.to(device)
    return P


class TripletFeatures(nn.Module):

    def __init__(self, M:int=41, rho=1, index_type='reduce-1'):
        super(TripletFeatures, self).__init__()
        self.M, self.L, self.rho, self.index_type = M, M - 1, rho, index_type
        self.index = self.get_index()
        self.hdim = len(self.index)
    
    def valid_index(self, m,n):
        if self.index_type == 'full':
            return abs(m*n) <= self.rho * self.L //2 and abs(m) + abs(n) <= self.L //2
        elif self.index_type == 'reduce-1':
            return (abs(m*n) <= self.rho * self.L //2) and (n >= m) and abs(m) + abs(n) <= self.L //2
        elif self.index_type == 'reduce-2':
            return (abs(m*n) <= self.rho * self.L //2) and (n >= abs(m)) and abs(m) + abs(n) <= self.L //2
        elif self.index_type == 'FWM':
            return (abs(m*n) <= self.rho * self.L //2) and (n >= abs(m)) and abs(m) + abs(n) <= self.L //2 and m*n != 0
    
    def get_index(self):
        '''
            Get symetric pertubation indexes.
            S = {(m,n)| |mn|<= rho*L/2, |m|<=L/2, |n|<= L/2, n>=|m|}
        '''
        S = [(m,n) for m in range(-self.L//2, self.L//2+1) for n in range(-self.L//2, self.L//2+1) if self.valid_index(m,n)]
        return S
    
    def nonlinear_features(self, E):
        Es = []
        p = self.M // 2
        if self.index_type == 'full':
            for m,n in self.index:
                Es.append(triplets(E, m,n))           # list of [batch, Nmodes]
        elif self.index_type == 'reduce-1':
            for m,n in self.index:
                if n == m:
                    Es.append(triplets(E,m,n))           # list of [batch, Nmodes]
                else:
                    Es.append(triplets(E,m,n) + triplets(E,n,m))
        else:
            for m,n in self.index:
                if n == m:
                    if m == 0:
                        Es.append(triplets(E, m,m))
                    else:
                        Es.append(triplets(E, m,m) + triplets(E, -m,-m))
                else:
                    if m+n == 0:
                        Es.append(triplets(E, m,n) + triplets(E, n,m))
                    else:
                        Es.append(triplets(E, m,n) + triplets(E, n,m) + triplets(E,-m, -n) + triplets(E, -n, -m))

        return torch.stack(Es, dim=-1)    # [batch, Nmodes, len(S)]
    


class eqPBC(nn.Module):

    def __init__(self, M:int=41, rho=1, index_type='reduce-1', pol_share=False):
        super(eqPBC, self).__init__()
        assert M % 2 == 1, 'M must be odd.'
        self.features = TripletFeatures(M, rho, index_type)
        self.pol_num = 1 if pol_share else 2
        self.nn = nn.ModuleList([ComplexLinear(self.features.hdim, 1, bias=False) for _ in range(self.pol_num)])
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
        Input:
            signal:  [batch, M, Nmodes] 
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        '''
        Nmodes = x.shape[-1]                    # [batch]
        if Nmodes == 1 and self.pol_num == 2: raise ValueError('Nmodes=1 and pol_num=2 is not a good choise, please set pol_share=True.')
        
        P = get_power(task_info, Nmodes, x.device)
        features = self.features.nonlinear_features(x)                                    # [batch, Nmodes, len(S)]  
        E = [self.nn[min(i, self.pol_num - 1)](features[...,i,:]) for i in range(Nmodes)]     # [batch, 1] 
        E = torch.cat(E, dim=-1)                                                          # [batch, Nmodes]
        return x[:,self.features.M//2,:] + E * P[:,None]                                  # [batch, Nmodes]

    


class eqAMPBC(nn.Module):
    '''
    Latest version of AmFoPBC. Nmodes=2.
    '''

    def __init__(self, M:int=41, rho=1, fwm_share=False):
        super(eqAMPBC, self).__init__()
        self.M, self.L, self.rho = M, M - 1, rho
        self.xpm_size, self.overlaps = M, M - 1
        self.fwm_modes = 1 if fwm_share else 2
        self.features = TripletFeatures(M, rho, index_type='FWM')

        self.C00 = nn.Parameter(torch.zeros(()), requires_grad=True)     # SPM coeff
        self.fwm_nn = nn.ModuleList([ComplexLinear(self.features.hdim, 1, bias=False) for _ in range(self.fwm_modes)])  # FWM coeff
        self.xpm_conv1 = nn.Conv1d(1, 1, M, bias=False)
        self.xpm_conv2 = nn.Conv1d(1, 1, M, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fwm_nn:
            nn.init.zeros_(layer.real.weight)
            nn.init.zeros_(layer.imag.weight)
        for conv in [self.xpm_conv1, self.xpm_conv2]:
            nn.init.zeros_(conv.weight)

    
    def zcv_filter(self, conv, x):
        '''
        zeros center vmap filter.
        x: real [B, L, Nmodes] -> real [B, L -  xpm_size + 1, Nmodes]
        '''
        B = x.shape[0]
        Nmodes = x.shape[-1]
        x = x.transpose(1,2)                            # x [B, Nmodes, L]
        x = x.reshape(-1, 1, x.shape[-1])               # x [B*Nmodes, 1, L]
        c0 = conv.weight[0,0, self.xpm_size//2]
        x = conv(x) - c0 * x[:,:,(self.overlaps//2):-(self.overlaps//2)]     # x [B*Nmodes, 1, L - xpm_size + 1]
        x = x.reshape(B, Nmodes, x.shape[-1])          # x [B, Nmodes, L - xpm_size + 1]
        x = x.transpose(1,2)                           # x [B, L - xpm_size + 1, Nmodes] 
        return x


    def IXIXPM(self, E):
        '''
            E: [batch, M, Nmodes]
        '''
        x = E * torch.roll(E.conj(),1, dims=-1)                                                     # x [B, M Nmodes]
        x = self.zcv_filter(self.xpm_conv2,x.real) + (1j)*self.zcv_filter(self.xpm_conv2, x.imag)   # x [B, M - xpm_size + 1, Nmodes]
        x = E[...,(self.overlaps//2):-(self.overlaps//2),:].roll(1, dims=-1) * x                    # x [B, M - xpm_size + 1, Nmodes] 
        return x[:,0,:] * (1j)  #   [B, Nmodes]
              

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
        Input:
            signal:  [batch, M, Nmodes] or [M, Nmodes]
            task_info: torch.Tensor or None. [B, 4] ot None.    [P,Fi,Fs,Nch]
        Output:
            TorchSignal.
            Nmodes = 1:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} E_{b, k+n, i} E_{b, k+m+n, i}^* E_{b, k+m, i} C_{m,n}
            Nmodes = 2:
                O_{b,k,i} = gamma P0^{3/2} * sum_{m,n} (E_{b, k+n, i} E_{b, k+m+n, i}^* +  E_{b, k+n, -i} E_{b, k+m+n, -i}^*) E_{b, k+m, i} C_{m,n}
        '''
       
        P = get_power(task_info, x.shape[-1], x.device)
        x = x*torch.sqrt(P[:,None,None])

        # IFWM term
        features = self.features.nonlinear_features(x)       # [batch, Nmodes, hdim]
        E = [self.fwm_nn[min(self.fwm_modes-1, i)](features[...,i,:]) for i in range(x.shape[-1])]     # [batch, 1] x Nmodes
        E = torch.cat(E, dim=-1)                             # [batch, Nmodes]
     
        # SPM + IXPM
        power = torch.abs(x)**2                              # [B, M, Nmodes]
        ps = 2*power + torch.roll(power, 1, dims=-1)         # [B, M, Nmodes]
        phi = self.C00 * power[:, self.M//2,:].sum(dim=-1, keepdim=True) + 2*self.zcv_filter(self.xpm_conv1, ps)[:,0,:] # [B, Nmodes]

        E = E + self.IXIXPM(x)                               # [batch, Nmodes]
        E = E + x[:,self.M//2,:]*torch.exp(1j*phi)           # [batch, Nmodes] 
        E = E / torch.sqrt(P[:,None])                        # [batch, Nmodes]
        return E



class eqPBC_step(nn.Module):

    def __init__(self, M:int=41, rho=1, index_type='reduce-1', pol_share=False):
        super(eqPBC_step, self).__init__()
        self.M, self.rho, self.index_type = M, rho, index_type
        self.PBC = eqPBC(M, rho, index_type, pol_share) 
        self.overlaps = self.M - 1
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        '''
        batch, L, Nmodes = x.shape
        x = x.unfold(1, self.M, 1)    # [batch, L - M + 1, Nmodes, M]
        x = x.permute(0, 1, 3, 2)      # [batch,L - M + 1, M, Nmodes]
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # [batch*(L - M + 1), M, Nmodes]
        x = self.PBC(x, task_info.view(batch, 1, -1).expand(batch, L - self.M +1, -1).reshape((batch*(L - self.M +1), -1)))      # [batch*(L - M + 1), Nmodes]
        x = x.reshape(batch, -1, x.shape[-1]) # [batch, L - M + 1, Nmodes]
        return x
    

class eqAMPBC_step(nn.Module):

    def __init__(self, M:int=41, rho=1, fwm_share=False):
        super(eqAMPBC_step, self).__init__()
        self.M, self.rho = M, rho
        self.PBC = eqAMPBC(M, rho, fwm_share) 
        self.overlaps = self.M - 1
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
        x: [batch, L, Nmodes]
        task_info: [batch, 4]
        --> [batch, L - M + 1, Nmodes]
        '''
        batch, L, Nmodes = x.shape
        x = x.unfold(1, self.M, 1)    # [batch, L - M + 1, Nmodes, M]
        x = x.permute(0, 1, 3, 2)      # [batch,L - M + 1, M, Nmodes]
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # [batch*(L - M + 1), M, Nmodes]
        x = self.PBC(x, task_info.view(batch, 1, -1).expand(batch, L - self.M +1, -1).reshape((batch*(L - self.M +1), -1)))      # [batch*(L - M + 1), Nmodes]
        x = x.reshape(batch, -1, x.shape[-1]) # [batch, L - M + 1, Nmodes]
        return x


class MultiStepPBC(nn.Module):

    def __init__(self, steps: int, M:int=41, rho=1, index_type='reduce-2', pol_share=False):
        super(MultiStepPBC, self).__init__()
        self.steps = steps
        self.M = M
        self.PBC_list = nn.ModuleList([eqPBC_step(M, rho, index_type, pol_share) for i in range(steps)])
        self.overlaps = sum([PBC.overlaps for PBC in self.PBC_list])
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
            x: [batch, L, Nmodes]
            task_info: [batch, 4]
            --> [batch, L - (M-1)*steps, Nmodes]
        '''
        for i in range(self.steps):
            x = self.PBC_list[i](x, task_info)
        return x
    
class MultiStepAMPBC(nn.Module):

    def __init__(self, steps: int, M:int=41, rho=1, index_type='reduce-2', fwm_share=False):
        super(MultiStepPBC, self).__init__()
        self.steps = steps
        self.M = M
        self.PBC_list = nn.ModuleList([eqAMPBC_step(M, rho, index_type, fwm_share) for i in range(steps)])
        self.overlaps = sum([PBC.overlaps for PBC in self.PBC_list])
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        '''
            x: [batch, L, Nmodes]
            task_info: [batch, 4]
            --> [batch, L - (M-1)*steps, Nmodes]
        '''
        for i in range(self.steps):
            x = self.PBC_list[i](x, task_info)
        return x
    

class eqCNNBiLSTM(nn.Module):
    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int=41, Nmodes=2, channels=244, kernel_size=10, hidden_size=113, num_layers=1, res_net=True):
        super(eqCNNBiLSTM, self).__init__()
        self.M = M
        self.res_net = res_net
        self.Nmodes = Nmodes
        self.conv1d = nn.Conv1d(in_channels=2*Nmodes, out_channels=channels, kernel_size=kernel_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(2*hidden_size*(M-kernel_size+1), Nmodes*2, bias=False)
        # nn.init.normal_(self.dense.weight, mean=0.0, std=1e-6)  # Adjust the mean and std as needed

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device) # [batch]
        x = x * torch.sqrt(P[:,None,None])            # [batch, M, Nmodes]
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
        if self.res_net: x = x0 + 0.1 * x
        x = x / torch.sqrt(P[:,None])            # [batch, Nmodes]
        return x


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
    
    def forward(self, x:torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device)
        x = x * torch.sqrt(P[:,None,None])            # [batch, M, Nmodes]
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
        x = x / torch.sqrt(P[:,None])            # [batch, Nmodes]
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
        # nn.init.normal_(self.dense.weight, mean=0.0, std=0.01)  # Adjust the mean and std as needed

    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        P = get_power(task_info, self.Nmodes, x.device)
        x = x * torch.sqrt(P[:,None,None])            # [batch, M, Nmodes]
        x0 = x[:, self.M //2,:]                  # Complex [B, M, Nmodes]  -> complex [B, Nmodes]
        x = torch.cat([x.real, x.imag], dim=-1)  # Complex [B, M, Nmodes]  -> float [B, M, Nmodes*2]

        x, _ = self.lstm(x)                       # float [B, M, Nmodes*2]  -> float [B, M, hidden_size*2]
        x = self.flatten(x)
        x = self.dense(x)                         # float [B, M*hidden_size*2] -> float [B, Nmodes*2]
        
        # convert to complex
        x = x.view(-1, self.Nmodes, 2)                 # float [B, Nmodes*2] -> float [B, Nmodes, 2]
        x = x[..., 0] + (1j)*x[..., 1]            # float [B, Nmodes, 2] -> complex [B, Nmodes]
        if self.res_net: x = x + x0
        x = x / torch.sqrt(P[:,None])            # [batch, Nmodes]
        return x


class eqAMPBCaddNN(nn.Module):
    def __init__(self, pbc_info, nn_info):
        super(eqAMPBCaddNN, self).__init__()
        self.pbc = eqAMPBC(**pbc_info)
        self.nn = eqCNNBiLSTM(**nn_info)
    
    def forward(self, x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
        c = x.shape[1] // 2

        return self.pbc(x[:,c-self.pbc.M//2:c+self.pbc.M//2+1,:], task_info) + self.nn(x[:,c-self.nn.M//2:c+self.nn.M//2+1,:], task_info) - x[:,x.shape[1]//2,:]

class eqID(nn.Module):

    '''
    Complex [B, M, Nmodes] -> Complex [B, Nmodes]
    '''
    def __init__(self, M:int, Nmodes=2):
        super(eqID, self).__init__()
        self.M = M

    def forward(self, x, task_info):

        return x[:,self.M//2,:]




def Test(net, dataloader):
    '''
    net: nn.Module
    dataloader: torch.utils.data.DataLoader
    '''
    net.eval()
    mse, power, ber, N = 0,0,0, len(dataloader)
    with torch.no_grad():
        for Rx, Tx, info in dataloader:
            Rx, Tx, info = Rx.cuda(), Tx.cuda(), info.cuda()
            PBC = net(Rx, info)
            mse += MSE(PBC, Tx).item()
            power += MSE(Tx, 0).item() 
            ber += np.mean(BER(PBC, Tx)['BER'])
    net.train()
    return {'MSE':mse/N, 'SNR': 10*np.log10(power/mse), 'BER':ber/N, 'Qsq': Qsq(ber/N)}

def write_log(writer, epoch, train_loss, metric):
    '''
    writer: SummaryWriter
    epoch: int
    train_loss: float
    metric: dict
    '''

    writer.add_scalar('Loss/train',  train_loss, epoch)
    writer.add_scalar('Loss/Test', metric['MSE'], epoch)
    writer.add_scalar('Metric/SNR', metric['SNR'], epoch)
    writer.add_scalar('Metric/BER', metric['BER'], epoch)
    writer.add_scalar('Metric/Qsq', metric['Qsq'], epoch)
    print(epoch, 'Train MSE:',  train_loss, 'Test MSE:', metric['MSE'],  'Qsq:', metric['Qsq'], flush=True)
    



if __name__ == '__main__':
    M = 41
    Nmodes = 2
    L = 100
    # x = torch.randn(10, M, Nmodes) + 1j*torch.randn(10, M, Nmodes)
    # task_info = torch.randn(10, 4)
    # eq = eqPBC_step(M, 1)
    # y1 = eq.PBC(x, task_info)
    # y = eq(x, task_info)
    # print(y1.shape, y.shape)
    # print(y1 - y[:,0])


    x = torch.randn(10, L, Nmodes) + 1j*torch.randn(10, L, Nmodes)
    task_info = torch.randn(10, 4)
    eq =  MultiStepPBC(2, M, 1)
    # eq = eqPBC_step(M, 1)
    y = eq(x, task_info)
    print(y.shape)


        
