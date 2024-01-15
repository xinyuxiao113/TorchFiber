import numpy as np, jax, jax.numpy as jnp, jax.random as rd, scipy.constants as const
from functools import partial
from .utils import calc_time
from .operator import fft,ifft,fftfreq,fftshift, get_omega, scan

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


def choose_dz_3order(freqspace: float, Pch_dBm: float, Nch: int, beta2: float, gamma: float, dz_max: float=0.5, eps: float=1e-3) -> float:
    '''
    Input:
        freqspace:[Hz]   Pch_dBm:[dBm]   beta2:[s^2/km]  gamma:[/W/km]  dz_max:[km]
    Output:
        dz [km]
    '''
    Bw = freqspace * Nch # [Hz]
    power = 1e-3*10**(Pch_dBm/10) * Nch  # [W]
    dz = (eps/(gamma * power * (2*np.pi*beta2*Bw**2)**2))**(1/3)
    dz = 2**int(np.log2(dz))
    return min(dz_max, dz)


def dispersionOp(h: float, Edim: int, beta2: float, alpha: float, Nfft: int, Fs: float) -> jax.Array:
    '''
        Input: 
            h[km], Edim: 2 or 3, beta2:[s^2/km]  alpha: [dB/km], Fs:[Hz]
        Output:
            (Nfft,1) or (1,Nfft,1)
    '''
    omega = get_omega(Fs, Nfft)            # (Nfft,), [Hz]
    kernel = jnp.exp(-(alpha/2)*(h) - 1j*(beta2/2)*(omega**2)*(h))
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
    return jnp.array([[jnp.cos(alpha), jnp.sin(alpha)*jnp.exp(1j*phi)], 
               [-jnp.sin(alpha)*jnp.exp(-1j*phi), jnp.cos(alpha)]])

def Delay(h, dbeta1, Nfft, Fs):
    '''
    Input:
        h:[km]  dbeta1:[s/km]   Fs:[Hz]
    Output: 
        T(omega): [2,2,Nfft]
    '''
    
    omega = get_omega(Fs, Nfft)
    return jnp.array([[jnp.exp(-1j*omega*dbeta1/2*h), jnp.zeros(Nfft)], 
                      [jnp.zeros(Nfft), jnp.exp(1j*omega*dbeta1/2*h)]])


def scatter_matrix(key, h, dbeta1, Nfft, Fs, z):
    # [2,2,Nfft]
    k1,k2 = rd.split(key)
    alpha = rd.uniform(k1)*2*jnp.pi
    phi = rd.uniform(k2)*2*jnp.pi
    R = SU2(alpha, phi)  # [2,2]
    T = Delay(h, dbeta1, Nfft, Fs) #[2,2,Nfft]
    return  R[...,None] * T


def PMD(E, T):
    '''
        E: [batch, Nfft, Nmodes] or [Nfft, Nmodes],  Nmodes = 2
        T: [2,2,Nfft]
    '''
    if E.ndim == 3:
        return jnp.einsum('bnp,ptn->bnt', E,T)
    elif E.ndim == 2:
        return jnp.einsum('np, ptn->nt', E,T)
    else:
        raise(ValueError)


def edfa(key, Ei, Fs, G=20, NF=4.5, Fc=193.1e12):
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
    noise    = jax.random.normal(key, Ei.shape, dtype=jnp.complex64) * np.sqrt(p_noise)
    return Ei * np.sqrt(G_lin) + noise


@calc_time
def manakov_ssf(tx_data, key, Ltotal, Lspan, hz, alpha, D, gamma, Fc, amp='edfa', NF=4.5, order=2, openPMD=False, Dpmd=3, Lcorr=0.1, **kwargs):      
    """
    Manakov model split-step Fourier (symmetric, dual-pol.)
    Input:
        tx_data: a dict with keys 'signal' and 'config'.  [batch, Nfft, Nmodes]  or [ Nfft, Nmodes]
        key: random seed.   
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
    # Channel parameters  
    Ei = tx_data['signal']
    Fs = tx_data['config']['Rs'] * tx_data['config']['sps']
    Bandwidth = Fs
    alpha0  = alpha/(10*np.log10(np.exp(1)))    # [1]
    beta2 = get_beta2(D, Fc)                    # [ps*s/nm]=[s^2/km]
    Nfft = Ei.shape[-2]
    dbeta1 = Dpmd / np.sqrt(2*Lcorr) * 1e-12    # [s/km]

    # Amplifier
    myEDFA = partial(edfa, Fs=Bandwidth, G=alpha*Lspan, NF=NF, Fc=Fc)

    # Nonlinear coeff.
    Gamma  = 8/9*gamma if Ei.shape[-1] == 2 else gamma

    # Calculate step length  in one span.
    Nspans = int(np.floor(Ltotal/Lspan))
    if type(hz) == int or  type(hz) == float:
        Nsteps = int(Lspan / hz)
        dz = jnp.ones(Nsteps) * hz
    else:
        Nsteps = len(hz)
        dz = hz

    z = 0 
    if order == 1:
        # @jax.jit  
        def one_step(carry , h):   # type: ignore
            Ei, z, key = carry
            k1,key = rd.split(key)
            # First linear step (frequency domain)
            Ei = fft(Ei, axis=-2)
            Ei = Ei * dispersionOp(h,Edim=Ei.ndim, beta2=beta2, alpha=alpha0, Nfft=Nfft, Fs=Fs)
            if openPMD:
                Ti = scatter_matrix(k1,h,dbeta1,Nfft,Fs,z)
                Ei = PMD(Ei, Ti)
            Ei = ifft(Ei, axis=-2)
            # Nonlinear step
            Ei = Ei * jnp.exp(-1j * Gamma* jnp.sum(Ei*jnp.conj(Ei), axis=-1)[..., None] * hz)
            return (Ei, z+h, key),None
        


        # @jax.jit
        def one_span(carry, _):  # type: ignore
            Ei, z, key = scan(one_step, carry,  dz,  length=Nsteps)[0] # TODO

            if amp =='edfa':
                key, key1 = jax.random.split(key)
                Ei = edfa(key1, Ei, Fs=Bandwidth, G=alpha*Lspan, NF=NF, Fc=Fc)
            elif amp =='ideal':
                Ei = Ei * jnp.exp(alpha0/2*Nsteps*hz)
            elif amp == None:
                Ei = Ei * jnp.exp(0)
            return (Ei, z, key), None
        
    elif order == 2:
        # @jax.jit
        def one_step(carry , h):
            Ei, z, key = carry
            k1,key = rd.split(key)

            # First linear step (frequency domain) 
            Ei = Ei * dispersionOp(h/2, Edim=Ei.ndim, beta2=beta2, alpha=alpha0, Nfft=Nfft, Fs=Fs)  

            # PMD step
            if openPMD:
                Ti = scatter_matrix(k1,h,dbeta1,Nfft,Fs,z)
                Ei = PMD(Ei, Ti)
            # Nonlinear step (time domain) Ei [batch, Nfft, Nmodes]
            Ei = ifft(Ei, axis=-2)
            Ei = Ei * jnp.exp(-1j * Gamma* jnp.sum(Ei*jnp.conj(Ei), axis=-1)[..., None] * hz)
            # Second linear step (frequency domain)
            Ei = fft(Ei, axis=-2)       
            Ei = Ei * dispersionOp(h/2, Edim=Ei.ndim, beta2=beta2, alpha=alpha0, Nfft=Nfft, Fs=Fs)
            return (Ei, z+h, key), None
        
        # @jax.jit
        def one_span(carry, _):
            Ei, z, key = carry
            Ei =  fft(Ei, axis=-2)
            Ei, z, key = scan(one_step, (Ei, z, key), dz,  length=Nsteps)[0]
            Ei = ifft(Ei, axis=-2)

            if amp =='edfa':
                key, key1 = jax.random.split(key)
                Ei = edfa(key1, Ei, Fs=Bandwidth, G=alpha*Lspan, NF=NF, Fc=Fc)
            elif amp =='ideal':
                Ei = Ei * jnp.exp(alpha0/2*Nsteps*hz)
            elif amp == None:
                Ei = Ei * jnp.exp(0)
            return (Ei, z, key), None
    else:
        raise(ValueError)


    Ech = scan(one_span, (Ei, z, key), None, length=Nspans)[0][0]
    config = {'key':key, 'Ltotal':Ltotal, 'Lspan':Lspan, 'hz':hz, 'alpha':alpha, 'D':D, 'gamma':gamma,'Fc':Fc, 'amp':amp, 'NF':NF, 'order':order, 'openPMD':openPMD, 'Dpmd':Dpmd, 'Lcorr':Lcorr, 'unit info':'Ltotal:[km]  Lspan:[km]  hz:[km]  alpha:[dB/km]  D:[ps/nm/km]  Fc:[Hz]  gamma:[1/W/km]  Dpmd:[s/sqrt(km)]  Lcorr:[km] NF:[dB]'}
    return {'signal':jax.device_get(Ech), 'config':config}
    

def phaseNoise(key, lw, Nsamples, Ts):
    
    sigma2 = 2*np.pi*lw*Ts    
    phi = jax.random.normal(key,(Nsamples,),jnp.float32) * jnp.sqrt(sigma2)
  
    return jnp.cumsum(phi)