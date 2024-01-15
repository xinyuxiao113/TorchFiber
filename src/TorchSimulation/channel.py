import numpy as np,  torch, scipy.constants as const
from functools import partial
from torch.fft import fft,ifft,fftfreq,fftshift

from .utils import calc_time

def get_omega(Fs:float, Nfft:int) -> torch.Tensor:
    ''' 
    get signal fft angular frequency.
    Input:
        Fs: sampling frequency. [Hz]
        Nfft: number of sampling points. 
    Output:
        omega:jnp.Array [Nfft,]
    '''
    return 2*np.pi*Fs*fftfreq(Nfft)


def get_beta2(D:float, Fc:float) -> float:
    '''
        Input:
            D:[ps/nm/km]    Fc: [Hz]
        Output:
            beta2:    [ps*s/nm]=[s^2/km]
    '''
    c_kms = const.c / 1e3                       # speed of light (vacuum) in km/s
    lamb  = c_kms / Fc                          # [km]
    beta2 = -(D*lamb**2)/(2*np.pi*c_kms)        # [ps*s/nm]=[s^2/km]
    return beta2


def choose_dz(freqspace: float, Lspan: float, Pch_dBm: float, Nch: int, beta2: float, gamma: float, dz_max:float=0.5, eps:float=1e-2) -> float:
    '''
    Input:
        freqspace:[Hz]   Pch_dBm:[dBm]   beta2:[s^2/km]  gamma:[/W/km]  dz_max:[km]
    Output:
        dz [km]
    '''

    Bw = freqspace * Nch # [Hz]
    power = 1e-3*10**(Pch_dBm/10) * Nch  # [W]
    dz = eps*200/(np.abs(beta2) * Bw**2 * gamma * power * Lspan)
    dz = 2**(int(np.log2(dz)))
    return min(dz_max, dz)



def dispersionOp(h: float, Edim: int, beta2: float, alpha: float, Nfft: int, Fs: int) -> torch.Tensor:
    '''
        Input: 
            h[km], Edim: 2 or 3, beta2:[s^2/km]  alpha: [dB/km], Fs:[Hz]
        Output:
            (Nfft,1) or (1,Nfft,1)
    '''
    omega = get_omega(Fs, Nfft)  # (Nfft,), [Hz]
    kernel = torch.exp(-(alpha/2)*(h) - 1j*(beta2/2)*(omega**2)*(h))
    if Edim == 3:
        return kernel[None,:,None]
    elif Edim == 2:
        return kernel[:,None]
    else:
        raise(ValueError)


def SU2(alpha, phi):
    '''
    out put shape: [2,2]
    '''
    return torch.tensor([[torch.cos(alpha), torch.sin(alpha)*torch.exp(1j*phi)], 
               [-torch.sin(alpha)*torch.exp(-1j*phi), torch.cos(alpha)]])

def Delay(h, dbeta1, Nfft, Fs):
    '''
    Input:
        h:[km]  dbeta1:[s/km]   Fs:[Hz]
    Output: 
        T(omega): [2,2,Nfft]
    '''
    
    omega = get_omega(Fs, Nfft)
    return torch.tensor([[torch.exp(-1j*omega*dbeta1/2*h), torch.zeros(Nfft)], 
                         [torch.zeros(Nfft), torch.exp(1j*omega*dbeta1/2*h)]])


def scatter_matrix(h, dbeta1, Nfft, Fs, z):
    # [2,2,Nfft]

    alpha = torch.rand() * 2*torch.pi
    phi = torch.rand() * 2*torch.pi
    R = SU2(alpha, phi)  # [2,2]
    T = Delay(h, dbeta1, Nfft, Fs) #[2,2,Nfft]
    return  R[...,None] * T


def PMD(E, T):
    '''
        E: [batch, Nfft, Nmodes] or [Nfft, Nmodes],  Nmodes = 2
        T: [2,2,Nfft]
    '''
    if E.ndim == 3:
        return torch.einsum('bnp,ptn->bnt', E,T)
    elif E.ndim == 2:
        return torch.einsum('np, ptn->nt', E,T)
    else:
        raise(ValueError)


def edfa(Ei, Fs=100e9, G=20, NF=4.5, Fc=193.1e12, device='cpu'):
    """
    Simple EDFA model

    :param Ei: input signal field [nparray]
    :param Fs: sampling frequency [Hz][scalar]
    :param G: gain [dB][scalar, default: 20 dB]
    :param NF: EDFA noise figure [dB][scalar, default: 4.5 dB]
    :param Fc: optical center frequency [Hz][scalar, default: 193.1e12 Hz]    

    :return: amplified noisy optical signal [nparray]
    """
    # assert G > 0, 'EDFA gain should be a positive scalar'
    # assert NF >= 3, 'The minimal EDFA noise figure is 3 dB'
    
    NF_lin   = 10**(NF/10)
    G_lin    = 10**(G/10)
    nsp      = G_lin*NF_lin / (2*(G_lin - 1))
    N_ase    = (G_lin - 1)*nsp*const.h*Fc
    p_noise  = N_ase*Fs    
    noise    = torch.randn(Ei.shape, dtype=torch.complex64, device=device) * np.sqrt(p_noise)
    return Ei * np.sqrt(G_lin) + noise


@calc_time
def manakov_ssf(tx_data, seed, Ltotal, Lspan, hz, alpha, D, gamma, Fc, amp='edfa', NF=4.5, order=2, openPMD=False, Dpmd=3, Lcorr=0.1, device='cpu'):      
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)
    Input:
        tx_data: a dict with keys 'signal' and 'config'.  [batch, Nfft, Nmodes]  or [ Nfft, Nmodes]
        seed: random seed.   
        Ltotal: total fiber length [km][default: 400 km]
        Lspan: span length [km][default: 80 km]
        hz: step-size for the split-step Fourier method [km][default: 0.5 km]
        alpha: fiber attenuation parameter [dB/km][default: 0.2 dB/km]
        D: chromatic dispersion parameter [ps/nm/km][default: 16 ps/nm/km]
        gamma: fiber nonlinear parameter [1/W/km][default: 1.3 1/W/km]
        Fc: carrier frequency [Hz] [default: 193.1e12 Hz]
        amp: 'edfa', 'ideal', or 'None. [default:'edfa']
        NF: edfa noise figure [dB] [default: 4.5 dB]    
        order: SSFM order. 1 or 2
        PMD: PMD effect. True or False
        Dpmd: PMD coeff.  [ps/sqrt(km)]
        Lcorr: fiber correlation length. [km]
    
    Output:
        transdata: a dict.
        'signal', 'config'
    '''
        E_z = -1/2*alpha*E + j beta2/2 E_tt - j gamma |E|^2E
    '''
    """

    # Set random seed
    torch.manual_seed(seed)

    # Channel parameters  
    Ei = tx_data['signal'].to(device)
    Fs = tx_data['config']['Rs'] * tx_data['config']['sps']
    Bandwidth = Fs
    alpha0  = alpha/(10*np.log10(np.exp(1)))    # [1]
    beta2 = get_beta2(D, Fc)                    # [ps*s/nm]=[s^2/km]
    Nfft = Ei.shape[-2]
    dbeta1 = Dpmd / np.sqrt(2*Lcorr) * 1e-12    # [s/km]
 
    # Linear Operator
    linOperator = partial(dispersionOp, Edim=Ei.ndim, beta2=beta2, alpha=alpha0, Nfft=Nfft, Fs=Fs)
    LinOp_hz = linOperator(hz).to(device)
    LinOp_half_dz = linOperator(hz/2).to(device)

    # Amplifier
    myEDFA = partial(edfa, Fs=Bandwidth, G=alpha*Lspan, NF=NF, Fc=Fc, device=device)

    # Nonlinear coeff.
    Gamma  = 8/9*gamma if Ei.shape[-1] == 2 else gamma

    # Calculate step length  in one span.
    Nspans = int(np.floor(Ltotal/Lspan))
    if type(hz) == int or  type(hz) == float:
        Nsteps = int(Lspan / hz)
        dz = torch.ones(Nsteps) * hz
    else:
        Nsteps = len(hz)
        dz = hz

    z = 0 

    # one order scheme
    if order == 1:
        for i in range(Nspans):
            for j in range(Nsteps):
                # linear step  (frequency domain)
                Ei = fft(Ei, axis=-2)
                Ei = Ei * LinOp_hz

                # PMD step  (frequency domain) 
                if openPMD:
                    Ti = scatter_matrix(hz, dbeta1, Nfft, Fs, z).to(device)
                    Ei = PMD(Ei, Ti)
                Ei = ifft(Ei, axis=-2)

                # nonlinear step  (time domain)
                Ei = Ei * torch.exp(-1j * Gamma* torch.sum(Ei*torch.conj(Ei), dim=-1)[..., None] * hz)

            if amp =='edfa':
                Ei = myEDFA(Ei)
            elif amp =='ideal':
                Ei = Ei * np.exp(alpha0/2*Nsteps*hz)
            elif amp == None:
                Ei = Ei

    elif order == 2:
        Ei =  fft(Ei, axis=-2)
        for i in range(Nspans):
            for j in range(Nsteps):
                # First linear step (frequency domain) 
                Ei = Ei * LinOp_half_dz

                # PMD step  (frequency domain) 
                if openPMD:
                    Ti = scatter_matrix(hz, dbeta1, Nfft, Fs, z)
                    Ei = PMD(Ei, Ti)

                # Nonlinear step (time domain) Ei [batch, Nfft, Nmodes]
                Ei = ifft(Ei, axis=-2)
                Ei = Ei * torch.exp(-1j * Gamma* torch.sum(Ei*torch.conj(Ei), dim=-1)[..., None] * hz)

                # Second linear step (frequency domain)
                Ei = fft(Ei, axis=-2)       
                Ei = Ei * LinOp_half_dz 

                z = z + hz

            if amp =='edfa':
                Ei = myEDFA(Ei)
            elif amp =='ideal':
                Ei = Ei * np.exp(alpha0/2*Nsteps*hz)
            elif amp == None:
                Ei = Ei
        Ei =  ifft(Ei, axis=-2)
    else:
        raise(ValueError)

    config = {'seed':seed, 'Ltotal':Ltotal, 'Lspan':Lspan, 'hz':hz, 'alpha':alpha, 'D':D, 'gamma':gamma,'Fc':Fc, 'amp':amp, 'NF':NF, 'order':order, 'openPMD':openPMD, 'Dpmd':Dpmd, 'Lcorr':Lcorr, 'unit info':'Ltotal:[km]  Lspan:[km]  hz:[km]  alpha:[dB/km]  D:[ps/nm/km]  Fc:[Hz]  gamma:[1/W/km]  Dpmd:[s/sqrt(km)]  Lcorr:[km] NF:[dB]'}
    return {'signal':Ei.to('cpu'), 'config':config}
    

def phaseNoise(lw, Nsamples, Ts):
    
    sigma2 = 2*np.pi*lw*Ts    
    phi = torch.randn(Nsamples, dtype=torch.float32) * np.sqrt(sigma2)
  
    return torch.cumsum(phi, dim=0)

if __name__ == '__main__':

    from .transmitter import simpleWDMTx
    tx_data = simpleWDMTx(123, batch=10, M=16, Nbits=400000, sps=16, Nch=5, Nmodes=1, Rs=32e9, freqspace=50e9, Pch_dBm=0, Ai=1, Vpi=2, Vb=-2, Ntaps=4096, roll=0.1, pulse_type='rc', device='cuda:0')
    transdata = manakov_ssf(tx_data, seed=123, Ltotal=2000, Lspan=80, hz=0.5, alpha=0.2, D=16, gamma=1.3, Fc=193.1e12, amp='edfa', NF=4.5, order=1, openPMD=False, Dpmd=3, Lcorr=0.1, device='cuda:0')

