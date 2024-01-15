import jax.numpy as jnp, jax.random as rd, pickle, jax,numpy as np, pandas as pd
import scipy.constants as const, scipy.special as special
from collections import namedtuple
from jax._src.config import config

from .transmitter import local_oscillator, phaseNoise, QAM, simpleWDMTx,circular_noise
from .operator import circFilter, L2, fft, ifft, fftfreq, get_omega
from .core import MySignal, parameters, SigTime


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
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    i1 = R*E1 * jnp.conj(E1)
    i2 = R*E2 * jnp.conj(E2)    

    return i1-i2

def hybrid_2x4_90deg(E1, E2):
    """
    Optical 2 x 4 90° hybrid
    
    :param E1: input signal field [nparray]
    :param E2: input LO field [nparray]
        
    :return: hybrid outputs
    """
    assert E1.size == E2.size, 'E1 and E2 need to have the same size'
    
    # optical hybrid transfer matrix    
    T = jnp.array([[ 1/2,  1j/2,  1j/2, -1/2],
                  [ 1j/2, -1/2,  1/2,  1j/2],
                  [ 1j/2,  1/2, -1j/2, -1/2],
                  [-1/2,  1j/2, -1/2,  1j/2]])
    
    Ei = jnp.array([E1, jnp.zeros((E1.size,)), jnp.zeros((E1.size,)), E2])    # [4, N]
    
    Eo = T@Ei
    
    return Eo

def coherentReceiver(Es, Elo, Rd=1):
    """
    Single polarization coherent optical front-end
    
    :param Es: input signal field [nparray]
    :param Elo: input LO field [nparray]
    :param Rd: photodiode resposivity [scalar]
    
    :return: downconverted signal after balanced detection    
    """
    assert Es.size == Elo.size, 'Es and Elo need to have the same size'
    
    # optical 2 x 4 90° hybrid 
    Eo = hybrid_2x4_90deg(Es, Elo)
        
    # balanced photodetection
    sI = balancedPD(Eo[1,:], Eo[0,:], Rd)
    sQ = balancedPD(Eo[2,:], Eo[3,:], Rd)
    
    return sI + 1j*sQ


def linFiberCh(Ei, L, alpha, D, Fc, Fs):
    """
    Linear fiber channel w/ loss and chromatic dispersion

    :param Ei: optical signal at the input of the fiber
    :param L: fiber length [km]
    :param alpha: loss coeficient [dB/km]
    :param D: chromatic dispersion parameter [ps/nm/km]   
    :param Fc: carrier frequency [Hz]
    :param Fs: sampling frequency [Hz]
    
    :return Eo: optical signal at the output of the fiber
    """
    #c  = 299792458   # speed of light [m/s](vacuum)    
    c_kms = const.c/1e3
    λ  = c_kms/Fc
    α  = alpha/(10*np.log10(np.exp(1)))
    β2 = -(D*λ**2)/(2*np.pi*c_kms)
    
    Nfft = len(Ei)

    ω = 2*np.pi*Fs*fftfreq(Nfft)
    ω = ω.reshape(ω.size,1)
    
    try:
        Nmodes = Ei.shape[1]
    except IndexError:
        Nmodes = 1
        Ei = Ei.reshape(Ei.size,Nmodes)

    ω = jnp.tile(ω,(1, Nmodes))
    Eo = ifft(fft(Ei,axis=0) * jnp.exp(-α*L - 1j*(β2/2)*(ω**2)*L), axis=0)
    
    if Nmodes == 1:
        Eo = Eo.reshape(Eo.size,)
        
        
    return Eo, jnp.exp(-α*L - 1j*(β2/2)*(ω**2)*L)



def rx(E: MySignal, chid:int, new_sps:int, Nch:int, freqspace:float) -> MySignal:
    ''' 
    Get single channel information from WDM signal.
    Input:
        E: 1D array. WDM signal. (Nfft,Nmodes) or (batch, Nfft, Nmodes)
        k: channel id.  [0,1,2,...,Nch-1]
        new_sps
    Output:
        E0: single channel signal. (Nfft,Nmodes)
    '''
    assert E.t.sps % new_sps == 0
    k = chid - Nch // 2
    Nfft = E.val.shape[-2]
    Fs = E.Fs
    t = jnp.linspace(0,1/Fs*Nfft, Nfft)
    omega = get_omega(E.Fs, Nfft)
    f = omega/(2*np.pi)  # [Nfft]
    if E.val.ndim == 2:
        H = (jnp.abs(f - k*freqspace)<freqspace/2)[:,None]
    elif E.val.ndim == 3:
        H = (jnp.abs(f - k*freqspace)<freqspace/2)[None, :,None]
    else:
        raise(ValueError)
    
    x0 = ifft(jnp.roll(fft(E.val, axis=-2) * H, -k*int(freqspace/Fs*Nfft), axis=-2), axis=-2)
    rate = E.t.sps // new_sps
    y = downsampling(x0, rate)
    newt = E.t.replace(sps=new_sps) # type: ignore
    
    return E.replace(val=y, t=newt, Fs=E.Fs/rate) # type: ignore


def decision(x, constSymb):
    '''
     x: ()
     constSymb: (M,)
    '''
    k = jnp.argmin(jnp.abs(x - constSymb)**2)
    return constSymb[k]


def nearst_symb(y, constSymb):
    '''
        y:[Nsymb, Nmodes]
        ConstSymb: [M]
    '''
    return jax.vmap(jax.vmap(decision, in_axes=(0, None),out_axes=0), in_axes=(-1, None),out_axes=-1)(y, constSymb)
    

def SER(y, truth):
    '''
        y:[Nsymb, Nmodes]
        ConstSymb: [M]
    '''
    const = jnp.unique(truth)
    z = nearst_symb(y, const)
    er = (z != truth)
    return jnp.mean(er, axis=0), z


def BER(y,truth,M=16, eval_range=(0,-1), **kwargs):
    '''
        y: [Nsymb,Nmodes]   L2(y) ~ 1
        truth: [Nsymb, Nmodes]   L2(truth) ~ 1

    return:
        metric, [Nbits, Nmodes]
    '''
    assert y.ndim == 2
    assert y.shape == truth.shape
    
    def getpower(x):
        return np.mean(np.abs(x)**2, axis=0)
    
    SNR_fn = lambda y, x: 10. * np.log10(getpower(x) / getpower(x - y))
    
    import scipy.special as special
    def Qsq(ber):
        return 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * ber), 0.))
    
    y = y[eval_range[0]:eval_range[1]]
    truth = truth[eval_range[0]:eval_range[1]]
    
    
    # SER
    ser,z = SER(y, truth)
    mod = QAM(M)
    bits = []
    data = []
    idx = []
    for i in range(z.shape[-1]):
        br = mod.demodulate(z[...,i]*np.sqrt(mod.Es), 'hard')
        bt = mod.demodulate(truth[...,i]*np.sqrt(mod.Es), 'hard')
        ber = np.mean(br!=bt)
        data.append([ber, ser[i], Qsq(ber), SNR_fn(y[:,i], truth[:,i])])
        bits.append(br)
        idx.append(f'dim{i}')
    metric = pd.DataFrame(data, columns=['BER', 'SER', 'Qsq', 'SNR'], index=idx)
    return metric
    # return metric, bits

def simpleRx(trans_data, tx_config, key, chid, rx_sps, FO=0, lw=0, phi_lo=0,Plo_dBm=10, method='frequency cut', **kwargs):
    '''
    Input:
        trans_data: [batch, Nfft, Nmodes] or [Nfft, Nmodes]
        tx_config: a dict with keys {'Rs', 'freqspace', 'pulse', 'Nch', 'sps'}
        key: rng for rx noise.
        chid: channel id.
        rx_sps: rx sps.
        FO: float. frequency offset. [Hz]
        lw: linewidth of LO.   [Hz]
        phi_lo: initial phase. [rads]
        Plo_dBm: Power of LO. [dBm].
    Output:
        signal, config.

    '''
    if type(tx_config) != dict:
        tx_config = tx_config.__dict__
        
    if 'sps' not in tx_config:
        tx_config['sps'] = tx_config['SpS']
    if 'freqspace' not in tx_config:
        tx_config['freqspace'] = tx_config['freqSpac']
    assert (trans_data.shape[-1] == 1 or trans_data.shape[-1] == 2)
        
        
    dims = trans_data.ndim
    N = trans_data.shape[-2]
    Ta = 1 / tx_config['Rs'] / tx_config['sps']
    freq = (chid - tx_config['Nch'] //2) * tx_config['freqspace']
    sigWDM = trans_data  # [batch, Nfft, Nmodes] or [Nfft, Nmodes]
    
    if dims == 2:
        sigLO, ϕ_pn_lo = local_oscillator(key,Ta,FO, lw,phi_lo,freq,N,Plo_dBm)
        # sigLO [Nfft], [Nfft]
    elif dims == 3:
        key_full = rd.split(key, num=trans_data.shape[0])
        sigLO, ϕ_pn_lo = jax.vmap(local_oscillator, in_axes=[0]+[None]*7, out_axes=0)(key_full,Ta,FO, lw,phi_lo,freq,N,Plo_dBm)
        # sigLo [batch, Nfft], [batch, Nfft]
    else:
        raise(ValueError)
    

    ## step 1: coherent receiver
    CR1 = jax.vmap(coherentReceiver, in_axes=(-1,None), out_axes=-1)
    if dims == 2:
        sigRx1 = CR1(sigWDM, sigLO)  # [Nfft, Nmodes], [Nfft]
    elif dims == 3:
        CR2 = jax.vmap(CR1, in_axes=(0,0), out_axes=0)
        sigRx1 = CR2(sigWDM, sigLO)  # [batch, Nfft, Nmodes], [batch, Nfft]
    else:
        raise(ValueError)
        

    # step 2: match filtering  
    if method == 'frequency cut':
        E = MySignal(val=sigRx1, t=SigTime(0,0,tx_config['sps']), Fs=1/Ta)
        sigRx2 = rx(E,chid,rx_sps,tx_config['Nch'],tx_config['freqspace']).val

    elif method == 'filtering':
        filter1 = jax.vmap(circFilter, in_axes=(None, -1), out_axes=-1)
        if dims == 2:
            sigRx2 = filter1(tx_config['pulse'], sigRx1)  
        elif dims == 3:
            filter2 = jax.vmap(filter1, in_axes=(None, 0), out_axes=0)
            sigRx2 = filter2(tx_config['pulse'], sigRx1)
        else:
            raise(ValueError)
        
        sigRx2 = downsampling(sigRx2, tx_config['sps'] // rx_sps)
    else:
        raise(ValueError)
        

    ## step 3: normalization and resampling # TODO: 可以优化！
    sigRx = sigRx2/L2(sigRx2)
    
    config = {'key':key,'chid':chid,'rx_sps':rx_sps,'FO': FO,'lw':lw,'phi_lo':phi_lo,'Plo_dBm':Plo_dBm,'method':method}
    return {'signal':sigRx, 'phase noise':ϕ_pn_lo, 'config':config}


def idealRx(trans_data, tx_config, key, chid, rx_sps, FO=0, lw=0, phi_lo=0,Plo_dBm=10, R=1, **kwargs):
    '''
    Input:
        key: rng for rx noise.
        E: WDM signal.  E.val [N, Nmodes]
        chid: channel id from [0,1,2,...,Nch-1].
        rx_sps: output sps.
        FO: float. frequency offset. [Hz]
        lw: linewidth of LO.  
    Output:
        sigRx: [Nsymb * rx_sps, pmodes]
        phi_pn: [Nsamples] noise.

    '''
    if 'sps' not in tx_config:
        tx_config['sps'] = tx_config['SpS']
    if 'freqspace' not in tx_config:
        tx_config['freqspace'] = tx_config['freqSpac']
        
    y = trans_data
    dims = trans_data.ndim
    Ta = 1 / tx_config['Rs'] / tx_config['sps']
    Plo = 10**(Plo_dBm/10)*1e-3

    ## step 0: WDM split
    E = MySignal(val=y, t=SigTime(0,0,tx_config['sps']), Fs=1/Ta)
    E1 = rx(E,chid,rx_sps,tx_config['Nch'],tx_config['freqspace'])

    ## step 1: phase noise
    N1 = E1.val.shape[-2]
    t  = jnp.arange(0, N1) * 1 / E1.Fs
    
    if dims == 2:
        phi = phi_lo + 2*np.pi*FO*t +  phaseNoise(key, lw, N1, 1/E1.Fs) # [Nfft]
        phi = circular_noise(phi)
    elif dims == 3:
        key_full = rd.split(key, num=trans_data.shape[0])
        phi = 2*np.pi*FO*t[None, :] +  jax.vmap(phaseNoise, in_axes=[0]+[None]*3, out_axes=0)(key_full, lw, N1, 1/E1.Fs)
        phi = jax.vmap(circular_noise, in_axes=0, out_axes=0)(phi)
        #  # [batch, Nfft]
    else:
        raise(ValueError)
    
    y = R*jnp.exp(Plo) * E1.val * jnp.exp(-1j*phi[...,None]) 
    y = y / L2(y)
    config = {'key':key,'chid':chid,'rx_sps':rx_sps,'FO': FO,'lw':lw,'phi_lo':phi_lo,'Plo_dBm':Plo_dBm}
    return {'signal':y, 'phase noise':phi, 'config':config}



def sml_dataset(sigRx, symbTx_, param_, paramCh_, paramRx_):
    '''
        generate dataset.
        Input:
            sigRx: [batch, N, pmodes] or [N,pmodes]
            symbTx_: The full channel symbols. [batch, Nsymb, channels,pmodes] or [Nsymb, channels,pmodes]
            param: Tx param.
            paramCh: channel param.
            paramRx: rx param.
        Output:
            DataInput = namedtuple('DataInput', ['y', 'x', 'w0', 'a'])
            y.shape [batch, Nfft, Nmodes] or [Nfft, Nmodes]
            x.shape  [batch, Nsymb, Nmodes] or [Nsymb, Nmodes]
    '''
    assert sigRx.ndim == symbTx_.ndim - 1
    param = param_.__dict__ if type(param_) != dict else param_
    paramCh = paramCh_.__dict__ if type(paramCh_) != dict else paramCh_
    paramRx = paramRx_.__dict__ if type(paramRx_) != dict else paramRx_

    if 'freqspace' not in param:
        Df = param['freqSpac']
    else:
        Df = param['freqspace']
    
    if sigRx.shape[-1] == 1:
        polmux = False
    else:
        polmux = True
        
    a = {'baudrate': param['Rs'],
    'channelindex': paramRx['chid'],
    'channels': param['Nch'],
    'distance': paramCh['Ltotal'] * 1e3,    # [m]
    'lpdbm': param['Pch_dBm'],    # [dBm]
    'lpw': 10**(param['Pch_dBm']/10)*1e-3, # [W]
    'modformat': str(param['M']) + 'QAM',
    'polmux': polmux,
    'samplerate': param['Rs'] * paramRx['rx_sps'],
    'spans': int(paramCh['Ltotal'] / paramCh['Lspan']),
    'srcid': 'src1',
    'D': paramCh['D'] * 1e-6,   #[s/m^2]
    'carrier_frequency': param['Fc'] + (paramRx['chid'] - param['Nch'] // 2) * Df,
    'fiber_loss': paramCh['alpha'] *1e-3, # [dB/m]
    'gamma': paramCh['gamma'] * 1e-3,    # [1/W/m]
    'sps': paramRx['rx_sps'] , 
    'M': param['M'],
    'CD': paramCh['D'] *  paramCh['Ltotal'] * 1e-3 ,  # cumulated dispersion    [s/m]
    'freqspace': Df,
    'unit information':'baudrate: Symbol rate [Hz].   channelindex: 0,1,2,...,Nch-1.   distance:[m]   lpdbm:[dBm]  lpw:[W]  samplerate:[Hz]  D:[s/m^2]  carrier_frequency:[Hz]  fiber_loss:[dB/m]  gamma:[1/W/m]'
    }

    if symbTx_.ndim == 4:
        symbTx = symbTx_[:,:,paramRx['chid']]
    elif symbTx_.ndim==3:
        symbTx = symbTx_[:,paramRx['chid']]
    else:
        raise(ValueError)

    # FO这里取了负号，get_data就不用了
    w0 = - 2 * np.pi * paramRx['FO'] / param['Rs']  # phase rotation each symbol.
    data_train_sml = DataInput(sigRx, symbTx,w0, a)
    return data_train_sml


def get_dataset(path_tx, path_rx, key, batch_id, chid, rx_sps, FO, lw, symb_only=True, method='CR frequency cut'):
    '''
    Get dataset from path.
    Input:
        path_tx: transmitter data path.
        path_rx: channel data path.
        key: receiver PRNGKey.
        batch_id: list or int or None. If None, use all the data.
        chid: channel id. 
        rx_sps: rx sps.
        FO: frequency offset. [Hz]
        lw: linewidth.        [Hz]
        symb_only: generate only symbol.(save time)
        method: 'CR frequency cut','CR filtering', 'ideal'
    Output:
        dataset. (y,x,w0,a)
    '''
    config.update("jax_enable_x64", True)     # float64 precision
    try:
        # x_batch:[batch, Nsymb, Nmodes]    symbWDM:[batch, Nsymb, Nch, Nmodes]
        with open(path_tx, 'rb') as file: x_batch, symbWDM, param = pickle.load(file)
        file.close()
        # y_batch: [batch, Nfft, Nmodes] or [batch, Nfft]  
        with open(path_rx, 'rb') as file: y_batch, paramCh = pickle.load(file)
        file.close()
        if (y_batch.shape[-1] > 2) and (y_batch.ndim == 2):
            y_batch = y_batch[...,None]  # [batch, Nfft, Nmodes]

        if batch_id == None:
            x = symbWDM
            y = y_batch
        else:
            assert np.max(batch_id) < y_batch.shape[0], f'batch_id must < {y_batch.shape[0]}'
            x = symbWDM[batch_id]        # [Nsymb, Nmodes]
            y = y_batch[batch_id]        # [Nfft, Nmodes]
    except:
        with open(path_tx,'rb') as file: param = pickle.load(file)
        file.close()
        tx_data = simpleWDMTx(symb_only, **param)
        with open(path_rx,'rb') as file: trans_data = pickle.load(file)
        file.close()
        paramCh = trans_data['config']
        if batch_id == None:
            x = tx_data['SymbTx']
            y = trans_data['signal']
        else:
            batch = param['batch']
            assert np.max(batch_id) < batch, f'batch_id must < {batch}'
            x = tx_data['SymbTx'][batch_id]
            y = trans_data['signal'][batch_id]
    if method=='CR frequency cut':
        rx_data = simpleRx(y, param, key, chid, rx_sps, FO, lw,method='frequency cut')
    elif method=='CR filtering':
        rx_data = simpleRx(y, param, key, chid, rx_sps, FO, lw,method='filtering')
    elif method == 'ideal':
        rx_data = idealRx(y, param, key, chid, rx_sps, FO, lw)
    else:
        raise(ValueError)
    data_sml = sml_dataset(jax.device_get(rx_data['signal']), jax.device_get(x), param, paramCh, rx_data['config'])
    config.update("jax_enable_x64", False)     # float32 precision
    return data_sml


def merge_dataset(data_list:list, center=0)->DataInput:
    '''
    merge dataset into one.
    Input:
        data_list: list of dataset.
        center:int, use this id's config dict.
    Out:
        Dataset. (y,x,w0,a)
    '''
    extend = lambda x: x[None,...] if x.ndim==2 else x
    get_batch = lambda x: x.shape[0] if x.ndim==3 else 1
    w0 = []
    y = []
    x = []
    lpdbm = []
    Nchs = []
    Fis = []
    Fss = []

    a = data_list[0].a.copy()

    for data in data_list:
        Fis = Fis + [data.a['carrier_frequency']] * get_batch(data.x)
        lpdbm = lpdbm + [data.a['lpdbm']] * get_batch(data.x)
        Nchs = Nchs + [data.a['channels']] * get_batch(data.x)
        Fss = Fss + [data.a['samplerate']] * get_batch(data.x)
        w0.append(data.w0)
        x.append(extend(data.x))
        y.append(extend(data.y))
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    w0 = np.array(w0)
    a['lpdbm'] = np.array(lpdbm)
    a['channels'] = np.array(Nchs)
    a['carrier_frequency'] = np.array(Fis)
    a['samplerate'] = np.array(Fss)

    return DataInput(y, x, w0, a)


def split_dataset(dataset):
    data_list = []
    batch = dataset.y.shape[0]
    
    for i in range(batch):
        a = dataset.a.copy()
        a['lpdbm'] = dataset.a['lpdbm'][i]
        a['channels'] = dataset.a['channels'][i]
        a['carrier_frequency'] = dataset.a['carrier_frequency'][i]
        a['samplerate'] = dataset.a['samplerate'][i]
        data = DataInput(dataset.y[i], dataset.x[i], dataset.w0[i], a)
        data_list.append(data)
    return data_list