import pickle, pandas as pd, torch, numpy as np
import scipy.constants as const, scipy.special as special
from collections import namedtuple
from torch.fft import fft, ifft, fftfreq, fftshift

from .transmitter import local_oscillator, phaseNoise, QAM, simpleWDMTx,circular_noise
from .channel import get_omega
from .utils import circFilter

## receive dataset structure.
'''
y: received signal. jax.Array with shape [batch, Nsymb*sps, Nmodes] or [Nsymb*sps, Nmodes]
x: transmitter symbols. jax.Array with shape [batch, Nsymb, Nmodes] or [Nsymb, Nmodes]
w0: init phase rotation each symbol. [rads/symbol]
a: information about the fiber system.
'''
DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])

def downsampling(x0, rate):
    if x0.ndim == 2:
        y = x0[::rate,:]
    elif x0.ndim == 3:
       y = x0[:,::rate,:]
    else:
        raise(ValueError)
    return y


def balancedPD(E1, E2, R=1):
    """
    Balanced photodetector (BPD)
    
    :param E1: input field [nparray]
    :param E2: input field [nparray]
    :param R: photodiode responsivity [A/W][scalar, default: 1 A/W]
    
    :return: balanced photocurrent
    """
    # assert R > 0, 'PD responsivity should be a positive scalar'
    assert E1.shape == E2.shape, 'E1 and E2 need to have the same size'
    i1 = R*E1 * torch.conj(E1)
    i2 = R*E2 * torch.conj(E2)    

    return i1-i2

def hybrid_2x4_90deg(E1, E2):
    """
    Optical 2 x 4 90° hybrid
    
    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]
        
    :return: hybrid outputs
    """
    assert E1.shape == E2.shape, 'E1 and E2 need to have the same size'
    
    # optical hybrid transfer matrix    
    T = torch.tensor([[ 1/2,  1j/2,  1j/2, -1/2],
                  [ 1j/2, -1/2,  1/2,  1j/2],
                  [ 1j/2,  1/2, -1j/2, -1/2],
                  [-1/2,  1j/2, -1/2,  1j/2]]).to(E1.device)
    
    Ei = torch.stack([E1, torch.zeros_like(E1), torch.zeros_like(E1), E2])    # [4, N]
    
    Eo = T@Ei.to(torch.complex64)
    
    return Eo

def coherentReceiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end
    
    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]
    
    :return: downconverted signal after balanced detection    
    """
    assert Es.shape == Elo.shape, 'Es and Elo need to have the same size'
    
    # optical 2 x 4 90° hybrid 
    Eo = hybrid_2x4_90deg(Es, Elo)
        
    # balanced photodetection
    sI = balancedPD(Eo[1,:], Eo[0,:], Rd)
    sQ = balancedPD(Eo[2,:], Eo[3,:], Rd)
    
    return sI + 1j*sQ




def rx(E: torch.Tensor, chid:int, sps_in:int,  sps_out:int, Nch:int, Fs: float, freqspace:float) -> torch.Tensor:
    ''' 
    Get single channel information from WDM signal.
    Input:
        E: 1D array. (Nfft,Nmodes) or (batch, Nfft, Nmodes)
        chid: channel id.
        sps_in: sps of input signal.
        sps_out: sps of output signal.
        Nch: number of channels.
        Fs: sampling rate.
        freqspace: frequency space between channels.
    Output:
        Eout: single channel signal. (Nfft,Nmodes)
    '''
    assert sps_in % sps_out == 0
    k = chid - Nch // 2
    Nfft = E.shape[-2]
    t = torch.linspace(0,1/Fs*Nfft, Nfft)
    omega = get_omega(Fs, Nfft)
    f = omega/(2*np.pi)  # [Nfft]
    if E.ndim == 2:
        H = (torch.abs(f - k*freqspace)<freqspace/2)[:,None]
    elif E.ndim == 3:
        H = (torch.abs(f - k*freqspace)<freqspace/2)[None, :,None]
    else:
        raise(ValueError)
    
    
    x0 = torch.fft.ifft(torch.roll(fft(E, axis=-2) * H.to(E.device), -k*int(freqspace/Fs*Nfft), dims=-2), dim=-2)
    Eout = downsampling(x0, sps_in // sps_out)
    
    return Eout


def nearst_symb(y, constSymb: torch.Tensor=QAM(16).const().to(torch.complex64)):    # type: ignore
    '''
        y: [*]
        ConstSymb: [M]
    '''
    constSymb = constSymb.to(y.device)
    const = constSymb.reshape([1]*len(y.shape) + [-1])   # type: ignore
    y = y.unsqueeze(-1) 
    k = torch.argmin(torch.abs(y - const), dim=-1)
    return constSymb[k]


def SER(y, truth):
    '''
        y:[batch, Nsymb, Nmodes] or [Nsymb, Nmodes]
        ConstSymb: [M]
    '''
    z = nearst_symb(y)
    er = torch.abs(z - truth) > 0.001
    return torch.mean(er * 1.0, dim=-2).to('cpu').numpy(), z


def _BER(y: torch.Tensor, truth: torch.Tensor, M=16, eval_range=(0,None)):
    '''
        y: [Nsymb,Nmodes]  or [batch, Nsymb,Nmodes]   L2(y) ~ 1
        truth: [Nsymb, Nmodes] or [batch, Nsymb,Nmodes]   L2(truth) ~ 1

    return:
        metric, [Nbits, Nmodes]
    '''
    assert y.ndim >= 2
    assert y.shape == truth.shape
    
    def getpower(x):
        return torch.mean(torch.abs(x)**2, dim=-2)
    
    SNR_fn = lambda y, x: 10. * torch.log10(getpower(x) / getpower(x - y))
    
    import scipy.special as special
    def Qsq(ber):
        return 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(np.minimum(2 * ber, 0.999)), 0.))
    
    y = y[..., eval_range[0]:eval_range[1], :]
    truth = truth[..., eval_range[0]:eval_range[1], :]
    
    
    # SER
    ser,z = SER(y, truth)
    mod = QAM(M)
    dim = z.ndim - 2

    br = mod.demodulate(z * np.sqrt(mod.Es), dim=dim)
    bt = mod.demodulate(truth * np.sqrt(mod.Es), dim=dim)

    ber = torch.mean((br!=bt)*1.0, dim=-2)

    return {'BER':ber.cpu().numpy(), 'SER':ser, 'Qsq':Qsq(ber.cpu().numpy()), 'SNR': SNR_fn(y, truth).cpu().detach().numpy()}


def BER(y: torch.Tensor, truth: torch.Tensor, M=16, eval_range=(0,None), batch=-1):
    '''
        Calculate BER.
            Input:
                y: [Nsymb,Nmodes]  or [batch, Nsymb,Nmodes]   L2(y) ~ 1
                truth: [Nsymb, Nmodes] or [batch, Nsymb,Nmodes]   L2(truth) ~ 1
                M: modulation order.    
                eval_range: range of symbols to evaluate.
                batch: batch size. -1 means all.   In order to prevent from OOM.
            Output: 
                metric, [Nbits, Nmodes]
    '''
    if y.ndim == 2 and batch!= -1: raise(ValueError)

    batch = y.shape[0] if batch == -1 else batch
    res = []
    for i in range(0, y.shape[0], batch):
        end = min(i+batch, y.shape[0])
        res.append(_BER(y[i:end], truth[i:end], M, eval_range))


    res = {k:np.concatenate([r[k] for r in res], axis=0) for k in res[0].keys()}
    return res


def L2(x: torch.Tensor)->torch.Tensor:
    '''
    Caculate average L2 Norm of x.
    Input:
        x: Array.
    Output:
        scaler.
    '''
    return torch.sqrt(torch.mean(torch.abs(x)**2))


def simpleRx(seed, trans_data, tx_config, chid, rx_sps, FO=0, lw=0, phi_lo=0,Plo_dBm=10, method='frequency cut', device='cuda:0'):
    '''
    Input:
        seed: random seed.
        trans_data: [batch, Nfft, Nmodes] or [Nfft, Nmodes]
        tx_config: a dict with keys {'Rs', 'freqspace', 'pulse', 'Nch', 'sps'}   with unit [Hz, Hz, Array, int, int] 
        chid: channel id.
        rx_sps: rx sps.
        FO: float. frequency offset. [Hz]
        lw: linewidth of LO.   [Hz]
        phi_lo: initial phase. [rads]
        Plo_dBm: Power of LO. [dBm].
        method: 'frequency cut' or 'filtering'.
    Output:
        signal, config.

    '''
    torch.manual_seed(seed)
    if type(tx_config) != dict:
        tx_config = tx_config.__dict__
        
    if 'sps' not in tx_config:
        tx_config['sps'] = tx_config['SpS']
    if 'freqspace' not in tx_config:
        tx_config['freqspace'] = tx_config['freqSpac']
    assert (trans_data.shape[-1] == 1 or trans_data.shape[-1] == 2)
        
        
    dims = trans_data.ndim
    batch = trans_data.shape[0]
    N = trans_data.shape[-2]
    Ta = 1 / tx_config['Rs'] / tx_config['sps']
    freq = (chid - tx_config['Nch'] //2) * tx_config['freqspace']
    sigWDM = trans_data.to(device)  # [batch, Nfft, Nmodes] or [Nfft, Nmodes]
    
    
    sigLO, ϕ_pn_lo = local_oscillator(batch, Ta, FO, lw, phi_lo, freq, N, Plo_dBm, device=device)
    

    ## step 1: coherent receiver
    CR1 = torch.vmap(coherentReceiver, in_dims=(-1,None), out_dims=-1)
    if dims == 2:
        sigRx1 = CR1(sigWDM, sigLO)  # [Nfft, Nmodes], [Nfft]
    elif dims == 3:
        CR2 = torch.vmap(CR1, in_dims=(0,0), out_dims=0)
        sigRx1 = CR2(sigWDM, sigLO)  # [batch, Nfft, Nmodes], [batch, Nfft]
    else:
        raise(ValueError)
        

    # step 2: match filtering  
    if method == 'frequency cut':

        sigRx2 = rx(sigRx1, chid, tx_config['sps'], rx_sps, tx_config['Nch'], 1/Ta, tx_config['freqspace'])

    elif method == 'filtering':
        filter1 = torch.vmap(circFilter, in_dims=(None, -1), out_dims=-1)
        if dims == 2:
            sigRx2 = filter1(tx_config['pulse'].to(device), sigRx1)  
        elif dims == 3:
            filter2 = torch.vmap(filter1, in_dims=(None, 0), out_dims=0)
            sigRx2 = filter2(tx_config['pulse'].to(device), sigRx1)
        else:
            raise(ValueError)
        
        sigRx2 = downsampling(sigRx2, tx_config['sps'] // rx_sps)
    else:
        raise(ValueError)
        

    ## step 3: normalization and resampling # TODO: 可以优化！
    sigRx = sigRx2/L2(sigRx2)
    
    config = {'seed':seed,'chid':chid,'rx_sps':rx_sps,'FO': FO,'lw':lw,'phi_lo':phi_lo,'Plo_dBm':Plo_dBm,'method':method}
    return {'signal':sigRx.to('cpu'), 'phase noise':ϕ_pn_lo, 'config':config}
