'''
    Baseline algorithms for digital signal processing.
        CDC: Chromatic dispersion compensation.
        DBP: Digital back propagation.
        DDLMS: Decision-directed least mean square.
        CDCDSP: CDC + DDLMS.
        DBPDSP: DBP + DDLMS.
        CDCtransform: CDC transform a signal_dataset.
        CDCDSP: CDC + DDLMS transform a signal_dataset.
        DBPDSP: DBP + DDLMS transform a signal_dataset.
'''
import torch, numpy as np
from torch.fft import fft,fftfreq,fftshift,ifft

from .dsp import ADF
from .core import TorchInput, TorchSignal, TorchTime
from .dataloader import signal_dataset
from .utils import calc_time


def get_omega(Fs:torch.Tensor, Nfft:int) -> torch.Tensor:
    ''' 
    get signal fft angular frequency.
    Input:
        Fs: sampling frequency. [Hz]  [batch,]
        Nfft: number of sampling points. 
    Output:
        omega:jnp.Array [batch, Nfft]
    '''
    return 2*torch.pi*Fs[:,None]*fftfreq(Nfft)[None,:].to(Fs.device)


def dispersion_kernel(dz:float, dtaps:int, Fs:torch.Tensor, beta2:float=-2.1044895291667417e-26, beta1:float=0, domain='time') -> torch.Tensor:
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
        h: torch.Tensor. (dtaps,)
        h is symmetric: jnp.flip(h) = h.
    '''
    omega = get_omega(Fs, dtaps)  # [batch, Nfft]
    kernel = torch.exp(-1j*beta1*omega*dz - 1j*(beta2/2)*(omega**2)*dz)  # [batch, Nfft]

    if domain == 'time':
        return fftshift(ifft(kernel, dim=-1),dim=-1)
    elif domain == 'freq':
        return kernel
    else:
        raise(ValueError)
    

def LinOp(E: torch.Tensor, z:float, dz: float, Fs: torch.Tensor,  beta2: float = -2.1044895291667417e-26, beta1: float = 0) -> torch.Tensor:
    ''' 
    Linear operator with time domain convolution.
    Input:
        E: E.val  [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        z: operator start position.  [m]
        dz: operator distance.      [m]
        Fs: samplerate, [Hz].  
        dtaps: kernel shape.
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    Nfft = E.shape[-2]
    kernel = dispersion_kernel(dz, Nfft, Fs, beta2, beta1, domain='freq')    # [batch, Nfft]
    kernel = kernel[...,None]       # [batch, Nfft, 1]
    x = ifft(fft(E, dim=-2) * kernel, dim=-2)

    if E.ndim == 2:
        x = x.squeeze(0)
    return x



def exp_integral(z: float, alpha:float = 4.605170185988092e-05, span_length:float=80e3) -> np.ndarray:
    '''
       Optical Power integral along z.
    Input:
        z: Distance [m]
        alpha: [/m]
        span_length: [m]
    Output:
        exp_integral(z) = int_{0}^{z} P(z) dz
        where P(z) = exp( -alpha *(z % Lspan))
    '''
    k = z // span_length
    z0 = z % span_length

    return k * (1 - np.exp(-alpha * span_length)) / alpha + (1- np.exp(-alpha * z0)) / alpha



def Leff(z: float, dz: float, alpha: float=4.605170185988092e-05, span_length: float=80e3):
    '''
       Optical Power integral along z.
    Input:
        z: Distance [m]
        dz: step length. [m]
        alpha: [/m]
        span_length: [m]
    Output:
        exp_integral(z) = int_{z}^{z + dz} P(z) dz
        where P(z) = exp( -alpha *(z % Lspan))
    '''
    return exp_integral(z + dz) - exp_integral(z)


def NonlinOp(E: torch.Tensor, z:float, dz:float, gamma=0.0016567) -> torch.Tensor:
    ''' 
    NonLinear operator.
    Input:
        E: E.val  [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        gamma: nonlinear coeff.  [/W/m]
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    phi = gamma * Leff(z, dz) * torch.sum(torch.abs(E)**2, dim=-1)[...,None]
    x = torch.exp(-(1j)*phi) * E # type: ignore
    return x


def CDC(E: torch.Tensor, Fs: torch.Tensor, length: float,  beta2: float = -2.1044895291667417e-26, beta1:float = 0) -> torch.Tensor:
    '''
        CD compensatoin.
    Input:
        E: digital signal.   [batch, Nfft,Nmodes]
        Fs: samplerate. [Hz]   [batch,]
        length >0, dz > 0: [m]
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]
    '''
    E = LinOp(E, length, -length, Fs, beta2, beta1)
    return E


def DBP(E: torch.Tensor, length: float, dz: float, Fs:torch.Tensor, power_dbm: torch.Tensor, beta2: float = -2.1044895291667417e-26, beta1:float = 0, gamma:float=0.0016567, order=1) -> torch.Tensor:
    '''
        Digital back propagation.
    Input:
        E: digital signal.          [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        Fs: samplerate, [Hz]            [1]       or [batch, 1]
        length >0, dz > 0: [m]
        dz: step size. [m]
        beta2: 2 order  dispersion coeff.     [s^2/m]
        beta1: 1 order dispersion coeff.      [s/m]
        power_dbm: power of each channel. [dBm]    per channel per mode power = 1e-3*10**(power_dbm/10)/Nmodes  [W].
    '''
    Nmodes = E.shape[-1]
    if Nmodes == 2: gamma = 8/9*gamma
    scale_param = 1e-3*10**(power_dbm/10)/Nmodes   # [batch]
    scale_param = scale_param.reshape([-1] + [1]*(E.ndim - 1))
    E = E * torch.sqrt(scale_param)
    K = int(length / dz)
    z = length 

    if order == 1:
        for i in range(K):
            E = LinOp(E, z, -dz, Fs, beta2, beta1)
            E = NonlinOp(E, z, -dz, gamma)
            z = z - dz
    elif order == 2:
        E = LinOp(E, z, -dz/2,  Fs, beta2, beta1)
        for i in range(K - 1):
            E = NonlinOp(E, z, -dz, gamma)
            E = LinOp(E, z, -dz, Fs, beta2,beta1)
            z = z - dz
        E = NonlinOp(E, z, -dz, gamma)
        E = LinOp(E, z, -dz/2, Fs, beta2, beta1)
        z = z - dz
    else:
        raise(ValueError)

    return E / torch.sqrt(scale_param)




def DDLMS(E: torch.Tensor, truth: torch.Tensor, sps:int, lead_symbols:int=2000, lr=[1/2**6, 1/2**7], taps=32) -> TorchSignal:
    '''
     DDLMS alg.  downsample input signal from sps to 1.
     Input:
        E: torch.Tensor with shape [B, L*sps, Nmodes].   The input signal.
        truth: torch.Tensor with shape [B, L, Nmodes].  truth signal.
        sps: samples per symbol of input signal.
        lead_symbols: number of pilot symbols before decision-directed.
    Output:
        compensated signal.
    '''
    signal_input = TorchSignal(E, TorchTime(0, 0, sps))
    signal_pilot = TorchSignal(truth, TorchTime(0, 0, 1))
    task_info = torch.zeros(E.shape[0], 4)

    meta_args = {'lr_init': lr}
    DDLMS = ADF(method='ddlms', taps=taps, Nmodes=E.shape[-1], batch_size=E.shape[0], mode='test', lead_symbols=lead_symbols, meta_args=meta_args)  # mode='test'很关键
    DDLMS.eval()
    DDLMS = DDLMS.to(E.device)
    with torch.no_grad():
        x = DDLMS(signal_input, signal_pilot, task_info)
    return x



def _CDCDSP(E: torch.Tensor, truth: torch.Tensor, length: float, Fs: torch.Tensor, sps: int, lead_symbols:int=2000) -> TorchSignal:
    '''
     CDC + DDLMS.
     Input:
        E: torch.Tensor with shape [B, L*sps, Nmodes].   The input signal.
        truth: torch.Tensor with shape [B, L, Nmodes].  truth signal.
        Fs: sample rate of E. [Hz]
        sps: samples per symbol of input signal E.
        lead_symbols: number of pilot symbols before decision-directed.
    Output:
        compensated signal.
    '''
    device = torch.cuda
    E = CDC(E, Fs, length)  # [batch, Nfft, Nmodes]
    F = DDLMS(E, truth, sps, lead_symbols)
    return F


def _DBPDSP(E: torch.Tensor, truth, length: float, dz: float, Fs:torch.Tensor, sps,  power_dbm: torch.Tensor, lead_symbols:int=2000,  beta2: float = -2.1044895291667417e-26, beta1:float = 0, gamma:float=0.0016567, order=1) -> TorchSignal:
    '''
     DBP + DDLMS.
     Input:
        E: torch.Tensor with shape [B, L*sps, Nmodes].   The input signal.
        truth: torch.Tensor with shape [B, L, Nmodes].  truth signal.
        dz: DBP step size. [m]
        Fs: sample rate of E. [Hz]
        sps: samples per symbol of input signal E.
        lead_symbols: number of pilot symbols before decision-directed.
    Output:
        compensated signal.
    '''
    E = DBP(E, length, dz, Fs, power_dbm, beta2, beta1, gamma, order)  # [batch, Nfft, Nmodes]
    x = DDLMS(E, truth, sps, lead_symbols)
    return x



def CDCtransform(dataset: signal_dataset, device='cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    CDC transform a signal_dataset.
    Output: y,x,info
    '''
    sps = dataset.data.a['sps']
    length = dataset.data.a['distance']
    res = []
    for data in dataset:
        E = data.signal_input.val.to(device)
        Fs = data.task_info[:,2].to(device)
        truth = data.signal_output.val.to(device)
        E = CDC(E, Fs, length)
        truth = data.signal_output.val
        res.append((E.to('cpu'), truth.to('cpu'), data.task_info.to('cpu')))
    
    return torch.cat([k[0] for k in res], dim=0), torch.cat([k[1] for k in res], dim=0), torch.cat([k[2] for k in res], dim=0)




@calc_time
def CDCDSP(dataset: signal_dataset, device='cpu', lead_symbols:int=2000) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    CDC+DDLMS transform a signal_dataset.
    Input:
        dataset: signal_dataset.
        device: 'cpu' or 'cuda:0'.
        lead_symbols: int. pilot numbers in DDLMS.
    Output: y,x,info
    '''
    sps = dataset.data.a['sps']
    length = dataset.data.a['distance']
    res = []
    for data in dataset:
        E = data.signal_input.val.to(device)
        Fs = data.task_info[:,2].to(device)
        truth = data.signal_output.val.to(device)
        E = _CDCDSP(E, truth, length, Fs, sps, lead_symbols)
        truth = data.signal_output.val[:, E.t.start : E.t.stop]
        res.append((E.val.to('cpu'), truth.to('cpu'), data.task_info.to('cpu')))
    
    return torch.cat([k[0] for k in res], dim=0), torch.cat([k[1] for k in res], dim=0), torch.cat([k[2] for k in res], dim=0)



@calc_time
def DBPDSP(dataset: signal_dataset, stps = 5, device='cpu', lead_symbols:int=2000):
    '''
    DBP+DDLMS transform a signal_dataset.
    Input:
        dataset: signal_dataset.
        stps: steps per span.
        device: 'cpu' or 'cuda:0'.
        lead_symbols: int. pilot numbers in DDLMS.
    Output: y,x,info
    '''
    sps = dataset.data.a['sps']
    length = dataset.data.a['distance']
    dz = length / (stps * dataset.data.a['spans'])

    res = []
    
    for data in dataset:
        Fs = data.task_info[:,2].to(device)
        power_dbm = data.task_info[:,0].to(device)

        E = data.signal_input.val.to(device)
        truth = data.signal_output.val.to(device)
        E = _DBPDSP(E, truth, length, dz, Fs, sps, power_dbm, lead_symbols)
        truth = data.signal_output.val[:, E.t.start : E.t.stop]
        res.append((E.val.to('cpu'), truth.to('cpu'), data.task_info.to('cpu')))
    
    return torch.cat([k[0] for k in res], dim=0), torch.cat([k[1] for k in res], dim=0), torch.cat([k[2] for k in res], dim=0)





if __name__ == '__main__':
    
    from src.TorchDSP.dataloader import signal_dataset, get_data

    test_data, info = get_data('data/test_data_few.pkl')
    test_dataset = signal_dataset(test_data, batch_size=test_data.y.shape[0], shuffle=False)

    predict, truth, task_info = CDCDSP(test_dataset, device='cuda:0', lead_symbols=1000)
    # predict, truth, task_info = DBPDSP(test_dataset, stps=1, device='cuda:0', lead_symbols=1000)
    print(predict.shape)

    
