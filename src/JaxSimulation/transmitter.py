import numpy as np, jax.numpy as jnp, jax
from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter, rcosfilter
from functools import partial
from .operator import circFilter,frame, circFilter_
from .utils import calc_time

def choose_sps(Nch:int, freqspace:float, Rs:float)->int:
    '''
    Choose transmitter sps.
    Input:
        Nch: Number of channels.circ
        freqspace: Frequency space. [Hz]
        Rs: Symbol rate. [Hz]
    Output:
        Transmitter samples per symbol.
    '''
    power = int(np.log2(Nch*freqspace/Rs))+1
    return 2**power


def circular_noise(noise):
    '''
    Modify a noise to 2pi circular.
    1D -> 1D.
    '''
    assert noise.ndim == 1
    N = noise.shape[0]
    df = (noise[-1] - noise[0]) % (2*jnp.pi)
    noise_new = noise - df * jnp.arange(N) / N
    return noise_new


def mzm(Ai, Vπ, u, Vb):
    """
    MZM modulator 
    
    :param Vπ: Vπ-voltage
    :param Vb: bias voltage
    :param u:  modulator's driving signal (real-valued)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    π  = np.pi
    Ao = Ai*jnp.cos(0.5/Vπ*(u+Vb)*π)
    
    return Ao


def iqm(Ai, u, Vπ, VbI, VbQ):
    """
    IQ modulator 
    
    :param Vπ: MZM Vπ-voltage
    :param VbI: in-phase MZM bias voltage
    :param VbQ: quadrature MZM bias voltage    
    :param u:  modulator's driving signal (complex-valued baseband)
    :param Ai: amplitude of the input CW carrier
    
    :return Ao: output optical signal
    """
    Ao = mzm(Ai/jnp.sqrt(2), Vπ, u.real, VbI) + 1j*mzm(Ai/jnp.sqrt(2), Vπ, u.imag, VbQ)
    
    return Ao

def signal_power(signal):
    """
    Compute the power of a discrete time signal.

    Parameters
    ----------
    signal : 1D ndarray
             Input signal.

    Returns
    -------
    P : float
        Power of the input signal.
    """
    return jnp.mean(jnp.abs(signal)**2)

def pulseShape(pulseType, SpS=2, N=1024, alpha=0.1, Ts=1):
    """
    Generate pulse shaping filters
    
    :param pulseType: 'rect','nrz','rrc'
    :param SpS: samples per symbol
    :param N: number of filter coefficients
    :param alpha: RRC rolloff factor
    :param Ts: symbol period
    
    :return filterCoeffs: normalized filter coefficients   

    pulse: [*,x1,x2,...,x2,x1]
    """  
    assert N % 2 == 0
    fa = (1/Ts)*SpS    # sample rate
    
    t = np.linspace(-2, 2, SpS)
    Te = 1       
    
    if pulseType == 'rect':
        filterCoeffs = np.concatenate((np.zeros(int(SpS/2)), np.ones(SpS), np.zeros(int(SpS/2))))       
    elif pulseType == 'nrz':
        filterCoeffs = np.convolve(np.ones(SpS), 2/(np.sqrt(np.pi)*Te)*np.exp(-t**2/Te), mode='full')        
    elif pulseType == 'rrc':
        tindex, filterCoeffs = rrcosfilter(N, alpha, Ts, fa)
    else :
        tindex, filterCoeffs = rcosfilter(N, alpha, Ts, fa)
        
    return filterCoeffs/np.sqrt(np.sum(filterCoeffs**2))


class QAM(QAMModem):
    '''
    QAM Modulation format.
    '''
    def __init__(self, M,reorder_as_gray=True):
        super(QAM, self).__init__(M)
        self.constellation_jnp = jnp.array(self.constellation)
        self.Es = jnp.array(self.Es)

    
    def bit2symbol(self, bits):
        N = len(bits)
        pow = 2**jnp.arange(N-1,-1,-1)
        idx = jnp.sum(pow*bits)
        return self.constellation_jnp[idx]
    
    def modulate(self, bits):
        bits_batch = frame(bits, self.num_bits_symbol, self.num_bits_symbol)
        symbol_batch = jax.vmap(self.bit2symbol)(bits_batch)
        return symbol_batch

    def const(self):
        return self.constellation / np.sqrt(self.Es)


@partial(jax.jit, static_argnums=1)
def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = jnp.empty(len(x) * n, dtype=x.dtype)
    y = y.at[0::n].set(x)
    return y


def phaseNoise(key, lw, Nsamples, Ts, dtype=jnp.float32):
    '''
    Generate phase noise.
    Input:
        key:PRNGKey.
        lw: linewidth. [Hz]
        Nsamples; number of samples.
        Ts: sample time. [s]
    Output:
        phase noise. (Nsamples,)
    '''
    σ2 = 2*np.pi*lw*Ts    
    phi = jax.random.normal(key,(Nsamples,),dtype) * jnp.sqrt(σ2)
  
    return jnp.cumsum(phi)



def local_oscillator(key, Ta, FO,  lw, phi_lo,  freq, N, Plo_dBm):
    '''
    key:PRNGKey.
    Ta: sample time. [s]
    FO: frequencey offset. [Hz]
    lw: linewidth.    [Hz]
    phi_lo: init phase error.  [rad]
    freq: frequency. [Hz]
    N: signal length. 
    Plo_dBm: power [dBm]
    '''
          
    Plo     = 10**(Plo_dBm/10)*1e-3            # power in W
    Δf_lo   = freq + FO              # downshift of the channel to be demodulated 
                                    
    # generate LO field
    π       = jnp.pi
    t       = jnp.arange(0, N) * Ta
    phi = phi_lo + 2*π*Δf_lo*t +  phaseNoise(key, lw, N, Ta)    # gaussian process
    phi = circular_noise(phi)
    sigLO   = jnp.sqrt(Plo) * jnp.exp(1j*phi)

    return sigLO, phi


def freq_grid(Nch: int, freqspace: float) -> jax.Array:
    '''
        generate a frequency grid.
    Input:
        Nch: number of channels.
        freqspace: frequency space.
    Output:
        jax.Array.
    '''
    freqGrid = jnp.arange(-int(Nch/2), int(Nch/2)+1,1) * freqspace
    if (Nch % 2) == 0:
        freqGrid += freqspace/2
    
    return freqGrid



def wdm_base(Nfft: int, fa: float, freqGrid: jax.Array) -> jax.Array:
    '''
        Generate a WDM waves.
    Input:
        Nfft: number of samples.
        fa: sampling rate. [hz]
        freqGrid: frequen
    Output:
        jax.Array with shape [Nfft, Nch].
    '''
    t = jnp.arange(0, Nfft)
    wdm_wave = jnp.exp(1j*2*jnp.pi/fa * freqGrid[None,:]*t[:,None]) # [Nsymb*SpS, Nch]
    return wdm_wave



@calc_time
def simpleWDMTx(symb_only,key, batch, M, Nbits, sps, Nch, Nmodes, Rs, freqspace, Pch_dBm=0, Ai=1, Vpi=2, Vb=-2, Ntaps=4096, roll=0.1, pulse_type='rc', **kwargs):
    """
    Simple WDM transmitter
    
    Generates a complex baseband waveform representing a WDM signal with arbitrary number of carriers
    symb_only: True or false.
    key: PRNG key.
    batch: number of signals.
    M: QAM order [default: 16]
    Nbits: total number of bits per carrier [default: 60000]
    sps: samples per symbol [default: 16]
    Nch: number of WDM channels [default: 5]
    Nmodes: number of polarization modes [default: 1]
    Rs: symbol rate. [Hz]
    freqspace: frequence space. [Hz]
    Pch_dBm: signal power per channel. [dBm]

    """
    # Verify sampling theorem
    fa = Rs * sps
    fc = Nch / 2 * freqspace
    print('Sample rate fa: %g, Cut off frequency fc: %g, fa > 2fc: %s' % (fa, fc, fa> 2*fc))
    if fa < 2*fc:
        print('sampling thm does not hold!')
        raise(ValueError)
    

    # modulation scheme
    mod = QAM(M=M)
    Es = mod.Es
    Pch = 10**(Pch_dBm/10)*1e-3

    # pulse shape
    pulse = pulseShape(pulse_type, sps, N=Ntaps, alpha=roll)

    # vmap scheme
    vmap = partial(jax.vmap, in_axes=(-1, None), out_axes=-1)

    # @jax.jit
    if symb_only == False:
        def one_channel(key, pulse): # type: ignore
            # step 1: generate random bits      bitsTx: [Nbits,]
            x = jax.random.randint(key, (Nbits,), 0, 2)
            # step 2: map bits to constellation symbols  symbTx: [Nsymb,]
            symbTx = mod.modulate(x)
            # step 3: normalize symbols energy to 1
            symbTx = symbTx/jnp.sqrt(Es)
            # step 4: upsampling   symbolsUp :  [Nsymb*SpS] d
            x = upsample(symbTx, sps)
            # step 5: pulse shaping
            x = circFilter_(pulse, x)
            # step 6: optical modulation
            x = iqm(Ai, 0.5*x, Vpi, Vb, Vb)
            # step 7: set the tx power [sqrt(W)] :   Pch [W] each channel
            x = jnp.sqrt(Pch/Nmodes) * x / jnp.sqrt(signal_power(x))
            return x, symbTx
    else:
        def one_channel(key, pulse):
            # step 1: generate random bits      bitsTx: [Nbits,]
            x = jax.random.randint(key, (Nbits,), 0, 2)
            # step 2: map bits to constellation symbols  symbTx: [Nsymb,]
            symbTx = mod.modulate(x)
            # step 3: normalize symbols energy to 1
            symbTx = symbTx/jnp.sqrt(Es)
            Nsymb = symbTx.shape[0]
            x = jnp.zeros([Nsymb*sps])
            return x, symbTx
    

    
    key_full = jax.random.split(key, batch*Nch*Nmodes).reshape(batch, 2, Nch, Nmodes) # type: ignore
    Tx = vmap(vmap(one_channel))
    
    signal_list = []
    symb_list = []
    for i in range(batch):
        sigWDM, SymbTx = Tx(key_full[i], pulse)             # [Nsymb*SpS, Nch, Nmodes]
        signal = wdm_merge(sigWDM, Rs*sps, Nch, freqspace)  # [Nsymb*SpS, Nmodes]
        signal_list.append(jax.device_get(signal))
        symb_list.append(jax.device_get(SymbTx))
        sigWDM, SymbTx = None, None
    signal = np.stack(signal_list, axis=0)
    symb = np.stack(symb_list, axis=0)

    sigWDM, SymbTx = jax.vmap(Tx, in_axes=(0, None), out_axes=0)(key_full, pulse)             # [batch, Nsymb*SpS, Nch, Nmodes]
    signal = wdm_merge(sigWDM, Rs*sps, Nch, freqspace)  # [Nsymb*SpS, Nmodes]

    config = {'key':key, 'batch':batch, 'M':16, 'Nbits':Nbits, 'sps':sps, 'Nch':Nch, 'Nmodes':Nmodes, 'Rs':Rs, 'freqspace':freqspace, 'Pch_dBm':Pch_dBm,'Ai':Ai, 'Vpi':Vpi, 'Vb':Vb, 'Ntaps':Ntaps, 'roll':roll, 'pulse_type':pulse_type, 'shape_info':'sigWDM:[batch, Nsymb*SpS, Nch, Nmodes],  SymbTx:[batch, Nsymb, Nch, Nmodes]', 'pulse':pulse, 'Fc':299792458/1550E-9}
    return {'signal':signal, 'SymbTx':symb, 'config':config}  #[batch, Nsymb*SpS, Nmodes]   [batch, Nsymb, Nch, Nmodes]



def wdm_merge(sigWDM: jax.Array, Fs: float, Nch: int, freqspace: float) -> jax.Array:
    '''
        Multiplex all WDM channel signals to a single WDM signal.
    Input:
        sigWDM: Signals for all WDM channels with shape (batch, Nfft, Nch, Nmodes) or [Nfft,Nch,Nmodes]
        Fs: float, sampling rate.
        Nch: number of channels.
        freqspace: frequency space.
    Output:
        The Single WDM signal with shape [batch, Nfft, Nmdoes] or [Nfft, Nmdoes].
    '''
    # E.shape  [batch, Nfft,Nch,Nmodes] or [Nfft,Nch,Nmodes]
    Nfft = sigWDM.shape[-3]
    freqGrid = jnp.arange(-int(Nch/2), int(Nch/2)+1,1) * freqspace

    wdm_wave = wdm_base(Nfft, Fs, freqGrid) # [Nfft, Nch]
    if sigWDM.ndim == 4:
        wdm_wave =  wdm_wave[None, ..., None]
    elif sigWDM.ndim == 3:
        wdm_wave = wdm_wave[..., None]
    else:
        raise(ValueError)

    x = jnp.sum(sigWDM * wdm_wave, axis=-2)
    return x  #[batch, Nfft, Nmdoes] or [Nfft, Nmdoes]




