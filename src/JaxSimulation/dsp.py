# type: ignore
import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt, jax.lax as lax
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,Union)
from flax.core import freeze, unfreeze,lift
from flax.linen.initializers import lecun_normal
from tqdm import tqdm
from collections import namedtuple
from functools import partial

from .operator import fft,ifft,fftfreq,fftshift, scan, dispersion_kernel, circFilter, Leff, frame, convolve, pretrain_kernel
from .receiver import BER, SER
from .transmitter import QAM
from .utils import calc_time, estimate_dtaps, get_dtaps, nn_vmap_batch, is_same_structure, get_beta1, get_beta2
from .core import MySignal, wrap_signal, conv1d_t, SigTime, conv1d_slicer, TrainState
from .adaptive_filter import piecewise_constant
from .layers import LRNN, fconv1d, cnn_block
from .initializers import delta, gauss, zeros, near_zeros
import src.JaxSimulation.adaptive_filter as af



def QAMrotation(z,symb,pilotInd):
    y = []
    for p in range(z.shape[-1]):
        ser_list = []
        for i in range(4):
            y1 = z[pilotInd,p] * jnp.exp(1j*i*np.pi/2)
            res = SER(y1[:,None],symb[pilotInd,p][:,None])[0]
            ser = res[0]
            ser_list.append(ser)

        i = np.argmin(ser_list)
        y.append(z[...,p] * jnp.exp(1j*i*np.pi/2))

    out = np.stack(y,axis=-1)
    return out


@partial(jax.jit, static_argnums=(1,3))
def bps(Ei, N, constSymb, B):
    """
    Blind phase search (BPS) algorithm.
    This algorithm can not back propagation. (piece wise constant, grad = 0)
    Input:
    ----------
    Ei : complex-valued ndarray
        Received constellation symbols. (Nsymb, Nmodes)
    N : int
        Half of the 2*N+1 average window.
    constSymb : complex-valued ndarray
        Complex-valued constellation.  (M,)
    B : int
        number of test phases.

    Output:
    -------
    Eo, theta, carry

    """
    phi = jnp.arange(0, B) * (jnp.pi / 2) / B - jnp.pi/4  # test phases  (B,)

    def Lk(phi, x, constSymb):
        '''
            () x () x (M,) -->  ()
        '''
        return jnp.min(jnp.abs(x*jnp.exp(1j*phi) - constSymb)**2)

    L_phi = jax.vmap(Lk, in_axes=(0,None,None),out_axes=0)          # (B,) x () x (M,) -->  (B,)
    L_phi_x = jax.vmap(L_phi, in_axes=(None,0,None),out_axes=1)     # (B,) x (Nsymb) x (M,)  --> (B,Nsymb)
    L_final = jax.vmap(L_phi_x, in_axes=(None,-1,None),out_axes=-1) # (B,) x (Nsymb, Nmodes) x (M,)  --> (B, Nsymb, Nmodes)

    Ls = L_final(phi, Ei, constSymb)   # (B,Nsymb,Nmodes)
    convolve_ = jax.vmap(partial(jnp.convolve, mode='same'), in_axes=(-1,None), out_axes=-1)  # (Nsymb,Nmodes) x (2N+1,) --> (Nsymb, Nmodes)
    convolve = jax.vmap(convolve_, in_axes=(0,None), out_axes=0)          # (B, Nsymb,Nmodes) x (2N+1,) --> (B, Nsymb, Nmodes)
    score = convolve(Ls, jnp.ones(2*N+1))  # (B, Nsymb, Nmodes)
    onehot = jax.nn.one_hot(jnp.argmin(score, axis=0),B).transpose([2,0,1]) # (B, Nsymb, Nmodes)
    phase = jnp.sum(onehot * phi[:,None,None], axis=0)
    theta = jnp.unwrap(phase, axis=0, period=np.pi/2)
    return Ei*jnp.exp(1j*theta), theta, None


def ddpll(Ei, Kv, constSymb, symbTx, carry=None, k0=1,k1=-1,k2=1, pilotInd=np.arange(200,dtype=int)):
    """
        Decision-directed Phase-locked Loop (DDPLL) algorithm.
        This algrithm can back propagation.
        # This algorithm is much faster on cpu !!
    Input:
        Ei : complex-valued ndarray. Received constellation symbols.
        Kv : float scalar. Loop filter gain.
        constSymb : complex-valued ndarra. Complex-valued ideal constellation symbols.
        symbTx : complex-valued ndarray. Transmitted symbol sequence.
        carry: initial phase estimate.
        k0,k1,k2: momentum parameters.
        pilotInd : int ndarray. Indexes of pilot-symbol locations.
    Output:
        Eo, theta, carry

    References
    [1] H. Meyer, Digital Communication Receivers: Synchronization, Channel 
    estimation, and Signal Processing, Wiley 1998. Section 5.8 and 5.9.    
    
    """
    Nsymb = Ei.shape[0]
    Nmodes = Ei.shape[1]
    # if symbTx == None:
    #     symbTx = jnp.zeros(Ei.shape)
    #     mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)
    # else:
    mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)


    # Loop filter coefficients  [1,k2,k2]
    a1b = jnp.array([k0, k1, k2])

    def Lk(phi, y, constSymb, target, mode):
        '''
            mode: pilot=False,  decision=True
        '''
        return jnp.min(jnp.abs(y*jnp.exp(1j*phi) - constSymb)**2) * mode + jnp.abs(y*jnp.exp(1j*phi) - target)**2 * (1-mode)

    g = jax.grad(Lk, argnums=0)

    def GD(carry, data):
        '''
            (3,), ()  x (),(),() --> (3,), () x ()
        '''
        u, phi = carry
        xk, yk, mode = data
        
        gk = g(phi, yk, constSymb, xk, mode)
        u = u.at[1].set(u[2])
        u = u.at[2].set(gk)
        u = u.at[0].set( jnp.sum(a1b * u))
        phip = phi
        phi = phi - Kv * u[0]
        return (u, phi),  phip
        # return jax.lax.stop_gradient((u, phi)),  jax.lax.stop_gradient(phip)
    GD_vmap = jax.vmap(GD, in_axes=-1, out_axes=-1)  # (3,Nmodes), (Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (3,Nmodes), (Nmodes) x (Nmodes)
    
    if carry == None:
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.zeros(Nmodes)
        carry = (u, phi)
    
    mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)
    data = (symbTx, Ei, mode)   # (Nsymb, Nmodes), (Nsymb, Nmodes), (Nsymb, Nmodes) 
    carry, theta = scan(GD_vmap, carry, data) # carry, phis
    theta = jnp.unwrap(theta, axis=0, period=np.pi/2)
    return Ei*jnp.exp(1j*theta), theta, carry


def foeddpll(Ei, Kf, Kn, constSymb, symbTx, carry=None, w0=0,p0=0,  k0=1,k1=-1,k2=1, pilotInd=np.arange(2000,dtype=int)):
    '''
        FOEddpll. remove two scale phase rotation!
     Input:
        Ei : complex-valued ndarray. Received constellation symbols.
        Kf : float scalar. learning rate for frequency offset (FO).
        Kn : float scalar. learning rate for phase rotation .
        constSymb : complex-valued ndarray. Complex-valued ideal constellation symbols.
        symbTx : complex-valued ndarray. Transmitted symbol sequence.
        carry: state. 
        w0,p0: intial FO, phase estimate.
        k0,k1,k2: momentum parameters.
        pilotInd : int ndarray. Indexes of pilot-symbol locations.

    Output:
        Eo, theta, carry
    '''

    Nsymb = Ei.shape[0]
    Nmodes = Ei.shape[1]
    mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)

    # Loop filter coefficients  [1,k2,k2]
    a1b = jnp.array([k0, k1, k2])

    def Lk(phi, y, constSymb, target, mode):
        '''
            mode: pilot=False,  decision=True
        '''
        return jnp.min(jnp.abs(y*jnp.exp(-1j*phi) - constSymb)**2) * mode + jnp.abs(y*jnp.exp(-1j*phi) - target)**2 * (1-mode)

    g = jax.grad(Lk, argnums=0)

    def GD(carry, data):
        '''
            (3,), (), ()  x (),(),() --> (3,), () x ()
        '''
        u, w, phi = carry
        xk, yk, mode = data
        
        phip = phi + w
        gk = g(phip, yk, constSymb, xk, mode)

        u = u.at[1].set(u[2])
        u = u.at[2].set(gk)
        u = u.at[0].set( jnp.sum(a1b * u))
        
        phi = phip - Kn * u[0]
        w = w - Kf * u[0]
        return (u, w, phi),  (phip, w)
        # return jax.lax.stop_gradient((u, phi)),  jax.lax.stop_gradient(phip)
    GD_vmap = jax.vmap(GD, in_axes=-1, out_axes=-1)  # (3,Nmodes), (Nmodes),(Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (3,Nmodes), (Nmodes) x (Nmodes)
    
    if carry == None:
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.ones(Nmodes) * (p0 % (2*np.pi))
        w = jnp.ones(Nmodes) * (w0 % (2*np.pi))
        carry = (u, w, phi)
    
    mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)
    data = (symbTx, Ei, mode)   # (Nsymb, Nmodes), (Nsymb, Nmodes), (Nsymb, Nmodes) 
    carry, (theta, w) = scan(GD_vmap, carry, data) # carry, phis
    theta = jnp.unwrap(theta, axis=0, period=np.pi/2)
    return Ei*jnp.exp(1j*theta), (theta, w), carry


def cpr(Ei, N, constSymb, symbTx,carry=None,pilotInd=jnp.arange(200,dtype=int)):    
    """
    Carrier phase recovery (CPR) for single mode.
    This algrithm can back propagation.
    Much faster on cpu.s

    return: Eo, theta, carry
    
    """    
    Nsymb = Ei.shape[0]
    Nmodes = Ei.shape[1]

    def D(x):
        d = jnp.abs(x - constSymb)**2
        idx = jnp.argmin(d)
        return constSymb[idx]


    def GD(carry, data):
        '''
        (N,), (), ()  x  (),(),() --> (N,), () x () 
        phi = [phi_k, phi_{k-1}, ..., phi_{k-N+1}]
        mode: True--use decision,  False--Use pilot. 
        '''
        phi, varphi, iter = carry
        xk,yk,mode = data
        pk = yk * jnp.exp(1j*varphi)
        dk = D(pk)*mode + xk*(1-mode)
        phik = jnp.angle(dk / pk) + varphi
        phi = jnp.roll(phi, 1)
        phi = phi.at[0].set(phik)
        
        varphi = (iter <= N) * jnp.mean(phik) + (iter > N) * jnp.mean(phi)
        
        iter = iter + 1
        return (phi, varphi, iter), varphi

    GD_vmap = jax.vmap(GD, in_axes=-1, out_axes=-1)  # (N,Nmodes), (Nmodes), (Nmodes)  x (Nmodes),(Nmodes),(Nmodes) --> (N,Nmodes), (Nmodes) x (Nmodes)
    if carry ==None:
        phi = jnp.zeros([N, Nmodes])
        varphi = jnp.mean(phi, axis=0)
        iter = jnp.zeros(Nmodes, dtype=int)
        carry = (phi, varphi, iter)
    mode = jnp.ones([Nsymb,Nmodes]).at[pilotInd].set(0)
    data = (symbTx, Ei, mode) 
    theta = scan(GD_vmap, carry, data)[1] # carry, phis
    return Ei*jnp.exp(1j*theta), theta, carry




@jax.jit
def LinOp(E:MySignal, z:float, dz: float, beta2: float = -2.1044895291667417e-26, beta1: int = 0) -> MySignal:
    ''' 
    Linear operator with time domain convolution.
    Input:
        E: E.val  [Nfft,Nmodes] or [batch, Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        dtaps: kernel shape.
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    Nfft = E.val.shape[-2]
    kernel = dispersion_kernel(dz, Nfft, jnp.mean(E.Fs), beta2, beta1, domain='freq') # [Nfft]
    kernel = kernel[:,None] if E.val.ndim == 2 else kernel[None,:,None]
    x = ifft(fft(E.val, axis=-2) * kernel, axis=-2)
    return E.replace(val=x)



@partial(jax.jit, static_argnums=(5,))
def LinOpTime(E:MySignal, z:float, dz: float, beta2: float = -2.1044895291667417e-26, beta1: float=0, dscale: float=1.0) -> MySignal:
    '''
        Linear operator with full FFT convolution.
    Input:
        E: E.val  [Nfft,Nmodes]
        z: operator start position.
        dz: operator distance.
        H: kernel function. [Nfft,]
    Output:
        E: E.val [Nfft, Nmodes]
    '''
    dtaps = estimate_dtaps(dz, E.Fs, beta2, dscale)
    kernel = dispersion_kernel(dz, dtaps, E.Fs, beta2, beta1, domain='time')
    x = jax.vmap(circFilter, in_axes=(None, -1), out_axes=-1)(kernel, E.val)
    return E.replace(val=x)



@jax.jit
def NonlinOp(E:MySignal, z:float, dz:float, gamma=0.0016567) -> MySignal:
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
    phi = gamma * Leff(z, dz) * jnp.sum(jnp.abs(E.val)**2, axis=-1)[...,None]
    x = jnp.exp(-(1j)*phi)*E.val
    return E.replace(val=x)

@calc_time
def BPS(signal: MySignal, truth:MySignal, pilotInd:np.array=np.arange(1000, dtype=int), N: int=51, B: int=121) -> MySignal:
    '''
    Blind phase search (BPS) algorithm with QAM rotation.(remove pi/2 ambiguous)
    z: [Nsymb, Nmodes]
    symb: [Nsymb, Nmodes]
    return :
    out, theta
    '''
    z = signal.val
    symb = truth.val
    mod = QAM(16)
    constSymb = mod.constellation / jnp.sqrt(mod.Es)
    z, theta,_ = bps(jax.device_get(z), N, jax.device_get(constSymb), B)

    # remove pi/2 ambiguous
    out = QAMrotation(z, symb, pilotInd)
    return signal.replace(val=out)



@calc_time
def DBP(E: MySignal, length: float, dz: float, power_dbm: float=0, beta2: float = -2.1044895291667417e-26, beta1:float = 0, gamma:float=0.0016567, order=1) -> MySignal:
    '''
        Digital back propagation.
    Input:
        E: digital signal.
        length >0, dz > 0: [m]
        power_dbm: power of each channel. [dBm]    per channel per mode power = 1e-3*10**(power_dbm/10)/Nmodes  [W].
    '''
    Nmodes = E.val.shape[-1]
    if Nmodes == 2: gamma = 8/9*gamma
    scale_param = 1e-3*10**(power_dbm/10)/Nmodes
    E = E.replace(val=E.val*np.sqrt(scale_param)) if E.val.ndim == 2 else E.replace(val=E.val*np.sqrt(scale_param)[:,None,None])
    K = int(length / dz)
    z = length 

    if order == 1:
        for i in range(K):
            E = LinOp(E, z, -dz,beta2,beta1)
            E = NonlinOp(E, z, -dz, gamma)
            z = z - dz
    elif order == 2:
        E = LinOp(E, z, -dz/2, beta2, beta1)
        for i in range(K - 1):
            E = NonlinOp(E, z, -dz, gamma)
            E = LinOp(E, z, -dz,beta2,beta1)
            z = z - dz
        E = NonlinOp(E, z, -dz, gamma)
        E = LinOp(E, z, -dz/2, beta2, beta1)
        z = z - dz
    else:
        raise(ValueError)

    return E.replace(val=E.val / np.sqrt(scale_param))


@calc_time
def TimeDBP(E: MySignal, length: float, dz: float, power_dbm: float=0, beta2: float = -2.1044895291667417e-26, beta1:float = 0, gamma:float=0.0016567, dscale=1.0) -> MySignal:
    '''
        Time domain Digital back propagation.
    Input:
        E: digital signal.
        length >0, dz > 0: [m]
        power_dbm: power of each channel. [dBm]    per channel per mode power = 1e-3*10**(power_dbm/10)/Nmodes  [W].
    '''
    Nmodes = E.val.shape[-1]
    if Nmodes == 2: gamma = 8/9*gamma
    scale_param = 1e-3*10**(power_dbm/10)/Nmodes
    E = E.replace(val=E.val*np.sqrt(scale_param))
    K = int(length / dz)
    z = length 
    for i in range(K):
        E = LinOpTime(E, z, -dz, beta2, beta1, dscale)
        E = NonlinOp(E, z, -dz, gamma)
        z = z - dz
    return E


@calc_time
def CDC(E: MySignal, length: float,  beta2: float = -2.1044895291667417e-26, beta1:float = 0) -> MySignal:
    '''
        CD compensatoin.
    Input:
        E: digital signal.
        length >0, dz > 0: [m]
        power_dbm: power of each channel. [dBm]    per channel per mode power = 1e-3*10**(power_dbm/10)/Nmodes  [W].
    '''
    E = LinOp(E, length, -length, beta2, beta1)
    return E


class conv1d(nn.Module):
    taps:int = 31
    rtap:Any =None
    mode:str='valid'          # 'full', 'same', 'valid'
    kernel_init:Callable=delta ##
    conv_fn: Callable=convolve

    @nn.compact
    def __call__(self,mysignal:MySignal) -> MySignal:
        '''
        mysignal.val: [Nfft, Nmodes]
        return: 
            'full' : [Nfft + taps  - 1]
            'valid': [Nfft - taps + 1]
            'same' : [Nfft]
        '''
        x = mysignal.val
        t = mysignal.t
        # t = self.variable('const', 't', conv1d_t, t, self.taps, self.rtap, 1, self.mode).value
        t = conv1d_t(t, self.taps, self.rtap, 1, self.mode)
        h = self.param('kernel', self.kernel_init, (self.taps,), jnp.complex64)
        x = self.conv_fn(x, h, mode=self.mode)

        return mysignal.replace(val=x, t=t)
    

class mimoaf(nn.Module):
    taps:int=32
    rtap:Any=None
    train: Any=False # 'train' or 'test'
    mimofn: Any=af.ddlms
    learnable: bool = True
    mimokwargs:Any=freeze({})
    mimoinitargs:Any=freeze({})
    
    @nn.compact
    def __call__(self, signal:MySignal, truth:MySignal, update_state:bool) -> MySignal:

        ## parameters
        if self.learnable:
            if self.mimofn == af.ddlms:
                eta_w = self.param('eta_w', lambda *_:jnp.arctanh(1/2**6))
                eta_f = self.param('eta_f', lambda *_:jnp.log(1/2**7))
                eta_s = self.param('eta_s', lambda *_:jnp.log(1/2**11))
                eta_b = self.param('eta_b', lambda *_:jnp.log(1/2**11))
                beta = self.param('beta', lambda *_:  jnp.log(1/2**11))
                lr_w = jax.nn.tanh(eta_w)
                lr_f = jnp.exp(eta_f)
                lr_s = jnp.exp(eta_s)
                lr_b = jnp.exp(eta_b)
                beta_ = jnp.exp(beta)
                mimo_fn = partial(self.mimofn,lr_w=lr_w, lr_f=lr_f, lr_s=lr_s, lr_b=lr_b, beta=beta_)
            elif self.mimofn == af.rde:
                eta = self.param('eta', lambda *_:jnp.arctanh(1/2**15))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(self.mimofn,lr=lr)
            elif self.mimofn == af.lms:
                eta = self.param('eta', lambda *_:jnp.arctanh(1e-4))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(self.mimofn,lr=lr)
            elif self.mimofn == af.mucma:
                eta = self.param('eta', lambda *_:jnp.arctanh(1e-4))
                beta = self.param('beta', lambda *_:jnp.arctanh(0.999))
                lr = jax.nn.tanh(eta)
                mimo_fn = partial(self.mimofn,lr=lr, dims=signal.val.shape[-1], beta=jax.nn.tanh(beta))
            else:
                raise(NotImplementedError)

        else:
            mimo_fn = self.mimofn

        x = signal.val
        dims = x.shape[-1]
        sps  = signal.t.sps
        t = conv1d_t(signal.t, self.taps, self.rtap, sps, 'valid')
        x = frame(x, self.taps, sps)  

        mimo_init, mimo_update, mimo_apply = mimo_fn(train=self.train, **self.mimokwargs)
        is_init = self.has_variable('state', 'mimoaf')     # 这个是为了初始化时，不更新carry.
        state = self.variable('state', 'mimoaf', lambda *_: (0, mimo_init(dims=dims, taps=self.taps, **self.mimoinitargs)), ())
        if truth is not None:
            truth = truth.val[t.start: truth.val.shape[-2] + t.stop]

        af_step, af_stats = state.value
        af_step, (af_stats, (af_weights, _)) = af.iterate(mimo_update, af_step, af_stats, x, truth)
        y = mimo_apply(af_weights, x) # 所有的 symbol 都过了之后才运用filter

        if update_state:  # TODO 
            state.value = (af_step, af_stats)
        
        return signal.replace(val=y, t=t, Fs=signal.Fs/sps)


class mimofoeaf(nn.Module):
    framesize:int=100
    w0:Any=0
    train:Any=False
    preslicer:Callable=lambda x: x
    foekwargs:Any=freeze({})
    mimofn:Callable=af.rde
    mimokwargs:Any=freeze({})
    mimoinitargs:Any=freeze({})

    @nn.compact
    def __call__(self, mysignal:MySignal, truth:MySignal) -> MySignal: 
        dims = mysignal.val.shape[-1]
        tx =  mysignal.t  # signal.val:[1090,2]   t:(450,-450,2)
        sps = mysignal.t.sps
        # MIMO
        ## slisig: [1030, 2], (480,-480,2)
        slisig = self.preslicer(mysignal)

        ## auxsig: [500,2], (247, -248, 1)
        auxsig = mimoaf(mimofn=self.mimofn,
                        train=self.train,
                        mimokwargs=self.mimokwargs,
                        mimoinitargs=self.mimoinitargs,
                        name='MIMO4FOE')(slisig, truth)
        # y [500, 2]  ty: (22,-23,1)
        y = auxsig.val # assume y is continuous in time
        ty = auxsig.t

        ## yf: [5,100,2]
        yf = frame(y, self.framesize, self.framesize)

        ## af.array 复制dims份ADF分别应用于不同dim, 作用的axis=-1， CPR中不再用到 truth
        Qs = self.param('Qs', lambda *_:jnp.array([[0,    0],
                                       [0, jnp.sqrt(1e-9)]]))
        Rs = self.param('Rs', lambda *_:jnp.array([[jnp.sqrt(1e-2), 0],
                                       [0, jnp.sqrt(1e-3)]]))
        alphas = self.param('alphas', lambda *_:jnp.arctanh(0.999))
        Q = Qs.T @ Qs 
        R = Rs.T @ Rs
        alpha = jax.nn.tanh(alphas)
        foe_init, foe_update, _ = af.array(af.frame_cpr_kf, dims)(Q=Q,R=R,alpha=alpha)
        state = self.variable('state', 'framefoeaf', lambda *_: (0., 0, foe_init(jnp.mean(self.w0))), ())

        # phi： float  af_step: int  af_stats: ((2, 1, 2), (2, 2, 2), (2, 2, 2)) = (z,P,Q)
        # phi:第一个符号应当旋转的角度（中间1000个）, af_stats的axis=-1为不同的极化方向 z=(theta, w)
        phi, af_step, af_stats = state.value

        # wf: [5,2] 收集了5个Block的 omega_k^-
        af_step, (af_stats, (wf, _)) = af.iterate(foe_update, af_step, af_stats, yf)  

        # wp: [5] 两个极化方向的数据几乎一致，这里直接取平均值
        wp = wf.reshape((-1, dims)).mean(axis=-1)

        # w:[1000]  framesize: 100
        w = jnp.interp(jnp.arange(y.shape[0] * sps) / sps, jnp.arange(wp.shape[0]) * self.framesize + (self.framesize - 1) / 2, wp) / sps
        # psi: [1000]
        psi = phi + jnp.cumsum(w)

        ## state value 更新
        state.value = (psi[-1], af_step, af_stats)

        # apply FOE to original input signal via linear extrapolation
        # psi_ext: [1090]
        psi_ext = jnp.concatenate([w[0] * jnp.arange(tx.start - ty.start * sps, 0) + phi,
                                psi,
                                w[-1] * jnp.arange(tx.stop - ty.stop * sps) + psi[-1]])

        signal = signal * jnp.exp(-1j * psi_ext)[:, None]
        return mysignal.replace(val=signal.val, t=signal.t)


class DDPLL(nn.Module):
    M:int=16
    mode: str='train'
    lead_symbols:int=20000
    learnable: bool=True
    @nn.compact
    def __call__(self, y: MySignal, x: MySignal) -> MySignal:
        '''
        Input: 
            y: MySignal. sps=1
            x: MySignal. sps=1
        '''
        Ei = y.val
        symbTx = x.val[y.t.start:x.val.shape[-2] + y.t.stop,:]
        if self.learnable:
            eta = self.param('ddpll eta', lambda *_:jnp.array(0.1))
        else:
            eta = jnp.array(0.1)
        k0 = jnp.arctanh(jnp.array(0.5))
        k1 = jnp.arctanh(jnp.array(-0.5))
        k2 = jnp.arctanh(jnp.array(0.5))   

        lr = jax.nn.tanh(eta)
        is_init = self.has_variable('state', 'ddpll')     # 这个是为了初始化时，不更新carry.
        carry = self.variable('state', 'ddpll', self.carry_init, Ei.shape[-1])
        constSymb = QAM(self.M).constellation / jnp.sqrt(QAM(self.M).Es)
        if self.mode == 'train':
            pilotInd = jnp.arange(Ei.shape[-2], dtype=int)  # Nsymb
        elif self.mode == 'test':
            pilotInd = jnp.arange(self.lead_symbols, dtype=int)  # Nsymb
        else:
            raise(ValueError)

        Eo, theta, carry_new = ddpll(Ei, lr, constSymb,symbTx, carry.value,k0=2*jax.nn.tanh(k0),k1=2*jax.nn.tanh(k1),k2=2*jax.nn.tanh(k2), pilotInd=pilotInd)

        if is_init:  # TODO 
            carry.value = carry_new
        return y.replace(val=Eo)

    def carry_init(self, Nmodes):
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.zeros(Nmodes)
        carry = (u, phi)
        return carry


class FOEDDPLL(nn.Module):
    M:int=16
    mode: str='train'
    lead_symbols:int=20000
    w0: float=0.0
    init_phase: float=0.0
    learnable: bool=True

    @nn.compact
    def __call__(self, y:MySignal, x:MySignal) -> MySignal:
        '''
        Input: 
            y: MySignal. sps=1
            x: MySignal. sps=1
        '''
        Ei = y.val
        symbTx = x.val[y.t.start:x.val.shape[-2] + y.t.stop,:]

        if self.learnable:
            eta_n = self.param('foeddpll eta_n', lambda *_:jnp.array(0.1))
            eta_f = self.param('foeddpll eta_f', lambda *_:jnp.array(0.01))
        else:
            eta_n = 0.1
            eta_f = 0.01
        k0 = jnp.arctanh(jnp.array(0.5))
        k1 = jnp.arctanh(jnp.array(-0.5))
        k2 = jnp.arctanh(jnp.array(0.5))
        Kf = jax.nn.tanh(eta_f)
        Kn = jax.nn.tanh(eta_n)
        is_init = self.has_variable('state', 'foeddpll')     # 这个是为了初始化时，不更新carry.
        carry = self.variable('state', 'foeddpll', self.carry_init, Ei.shape[-1])
        constSymb = QAM(self.M).constellation / jnp.sqrt(QAM(self.M).Es)
        if self.mode == 'train':
            pilotInd = jnp.arange(Ei.shape[-2], dtype=int)  # Nsymb
        elif self.mode == 'test':
            pilotInd = jnp.arange(self.lead_symbols, dtype=int)  # Nsymb
        else:
            raise(ValueError)

        Eo, theta, carry_new = foeddpll(Ei, Kf, Kn, constSymb,symbTx, carry.value,k0=2*jax.nn.tanh(k0),k1=2*jax.nn.tanh(k1),k2=2*jax.nn.tanh(k2), pilotInd=pilotInd)

        if is_init:  # TODO 
            carry.value = carry_new
        return y.replace(val=Eo)

    def carry_init(self, Nmodes):
        u = jnp.zeros([3, Nmodes])  # [u_f, u_d1, u_d]
        phi = jnp.ones(Nmodes) * (self.init_phase % (2*np.pi))
        w = jnp.ones(Nmodes) * (self.w0 % (2*np.pi))
        carry = (u, w, phi)
        return carry


class batch_state(nn.Module):
    @nn.compact
    def __call__(self, x:MySignal)->MySignal:
        batch_id = self.variable('state','batch id', lambda *_:0)
        batch_id.value = batch_id.value + 1
        return x




class Linear(nn.Module):
    '''
        Linear dispersion module.
        Input:
            dz: [m]
            dtaps: int
            beta2: [s^2/m]  float=-2.1044895291667417e-26
            beta1: [s/m]
            conv_fn: Callable.
    '''
    dtaps: int=41
    kernel_type: str = 'physical'        #  'physical', 'conv', ...
    kernel_init: Callable = delta
    dz: float=80e3                       # [m]
    beta2: float=-2.1044895291667417e-26 #[s^2/m]
    beta1: float=0.0                     # [s/m] 


    conv_fn: Callable=partial(convolve, mode='valid')
    @nn.compact
    def __call__(self, x:MySignal) -> MySignal:
        '''
            [Nfft, Nmodes] -> [Nfft - dtaps + 1, Nmodes]
        '''
        assert x.val.shape[0] > self.dtaps, f'x.val.shape[0] must be larger than self.dtaps'
        Nmodes = x.val.shape[-1]
        
        if self.kernel_type == 'conv':
            h = self.param('dispersion kernel',self.kernel_init, Nmodes) 
        elif self.kernel_type == 'physical':
            kernel = dispersion_kernel(self.dz, self.dtaps, x.Fs, self.beta2, self.beta1)
            h = jnp.stack([kernel]*Nmodes, axis=-1)
        else:
            raise(ValueError)

        y = jax.vmap(self.conv_fn, in_axes=-1, out_axes=-1)(h, x.val)
        t = conv1d_t(x.t, self.dtaps, None, 1, 'valid') 
        return x.replace(val=y, t=t)


from .layers import fun1d


class Nonlinear(nn.Module):
    '''
    Nonlinear module.
    Input:
        dz: [m]
        ntaps: int
        gamma:float= 0.0016567 # [/W/m]
        init_type: 'zeros', 'delta', 'gauss'
    '''
    batch_size: int=200
    ntaps:Any=1    # or tuple of int
    nchs: Any=()
    activation:Any=jax.nn.silu
    hidden_dim:Any=None
    GRU_depth: int=1
    kernel_init: Callable=lecun_normal()
    meta: str='Filter'  # 'CNN', 'GRU', 'ConvGRU', 'Filter', 'Fix'

    dz:float=80e3  #[m]
    gamma:float= 0.0016567 # [/W/m]
    @nn.compact
    def __call__(self,x: MySignal) -> MySignal:
        '''
            [Nfft, Nmodes] -> [Nfft - ntaps + 1, Nmodes]
        '''
        assert x.val.shape[0] > np.sum(self.ntaps), f'x.val.shape[0] must be larger than self.ntaps'
        sps = x.t.sps
        Nmodes = x.val.shape[-1]
        if self.meta == 'CNN':
            assert type(self.ntaps) == tuple
            shapes = self.ntaps
            chs = self.nchs + (Nmodes,)
            K = np.sum(np.array(shapes) - 1) + 1
            #t = self.variable('const','t',conv1d_t, x.t, K, None, 1, 'valid').value
            t = conv1d_t(x.t, K, None, 1, 'valid') 
            phi = cnn_block(kernel_shapes=shapes, channels=chs, param_dtype=jnp.float32, dtype=jnp.float32, padding='valid', activation=self.activation, n_init=self.kernel_init)(jnp.abs(x.val)**2)  # [N1, Nmodes]

        elif self.meta == 'GRU':
            t = conv1d_t(x.t, self.ntaps, None, 1, 'valid')
            hidden_dim = Nmodes if (self.hidden_dim == None ) else self.hidden_dim
            gru_dims= (hidden_dim,)*(self.GRU_depth - 1) + (Nmodes,)
            RNN = LRNN(depth=self.GRU_depth, dims=gru_dims)

            P = jnp.abs(x.val)**2
            SPM = fconv1d(features=Nmodes, kernel_size=(self.ntaps,),strides=(1,), kernel_init=self.kernel_init, param_dtype=jnp.float32, dtype=jnp.float32, padding='valid')(P) # [N, hidden_dim] 
            is_init = self.has_variable('state', 'nonlinear state')     # 这个是为了初始化时，不更新carry.
            h = self.variable('state', 'nonlinear state', lambda *_:RNN.carry_init())
            carry, XPM0 = RNN(h.value, P[0:self.batch_size*sps])
            _, XPM1 = RNN(carry, P[self.batch_size*sps:])
            XPM = jnp.concatenate([XPM0,XPM1], axis=0)
            phi = SPM + XPM[self.ntaps//2:-(self.ntaps//2)]
            if is_init:
                h.value = carry
        elif self.meta == 'ConvGRU':
            t = conv1d_t(x.t, self.ntaps, None, 1, 'valid')
            hidden_dim = Nmodes if (self.hidden_dim == None ) else self.hidden_dim
            gru_dims= (hidden_dim,)*(self.GRU_depth - 1) + (Nmodes,)
            RNN = LRNN(depth=self.GRU_depth, dims=gru_dims)

            P = jnp.abs(x.val)**2
            SPM = fconv1d(features=hidden_dim, kernel_size=(self.ntaps,),strides=(1,), kernel_init=self.kernel_init, param_dtype=jnp.float32, dtype=jnp.float32, padding='valid')(P) # [N, hidden_dim] 
            is_init = self.has_variable('state', 'nonlinear state')     # 这个是为了初始化时，不更新carry.
            h = self.variable('state', 'nonlinear state', lambda *_:RNN.carry_init())
            carry, PM0 = RNN(h.value, SPM[0:self.batch_size*sps])
            _, PM1 = RNN(carry, SPM[self.batch_size*sps:])
            phi = jnp.concatenate([PM0,PM1], axis=0)

            if is_init:
                h.value = carry
        
        elif self.meta == 'Filter':
            t = conv1d_t(x.t, self.ntaps, None, 1, 'valid')
            phi = fconv1d(features=Nmodes, kernel_size=(self.ntaps,),strides=(1,), kernel_init=self.kernel_init, param_dtype=jnp.float32, dtype=jnp.float32, padding='valid')(jnp.abs(x.val)**2) # [N, hidden_dim]
        elif self.meta == 'scaler':
            t = conv1d_t(x.t, self.ntaps, None, 1, 'valid')
            
            # nlkernel = jax.scipy.special.exp1(a*jnp.linspace(-1,1,self.ntaps)[:,None]**2/(x.Fs/160e9)**4)   # [ntaps, Nmodes*Nmodes]
            # nlkernel = fun1d(dims=Nmodes**2, m=200)(jnp.abs(jnp.linspace(-1,1,self.ntaps))[:,None])                     # [ntaps, Nmodes*Nmodes]
            # nlkernel = fun1d(dims=Nmodes**2, m=200)(jnp.abs(jnp.linspace(-1,1,self.ntaps))[:,None]**2)                     # [ntaps, Nmodes*Nmodes]

            # a = self.param('scaler', lambda *_:jnp.zeros(Nmodes))
            # width = self.param('nlkernel', lambda *_: jnp.ones(1))                       # [ntaps, Nmodes*Nmodes]
            # nlkernel = a*jnp.exp(-width * jnp.abs(jnp.linspace(-1,1,self.ntaps))[:,None])                # [ntaps, Nmodes*Nmodes]


            task_info = jnp.array([x.power, x.Nch / 10, x.Fs / 100e9])
            nlkernel = TaskMLP(features=self.ntaps*Nmodes**2)(task_info)                                      # [ntaps*Nmodes*Nmodes]

            nlkernel = jnp.reshape(nlkernel, (self.ntaps, Nmodes, Nmodes)).transpose((1,2,0))                 # [Nmodes, Nmodes, ntaps]
            inp = (jnp.abs(x.val)**2).transpose((1,0))[None, ...]                                             # [1, Nmodes, Nfft]
            phi = jax.lax.conv(inp, nlkernel, (1,), 'VALID')                                                  # [1, Nmodes, Nfft - ntaps + 1]
            phi = phi[0].transpose((1,0))                                                                     # [Nfft - ntaps + 1, Nmodes]
        elif self.meta == 'Fix':
            t = x.t
            phi = 0
        else:
            raise(ValueError)

        # phi = (|E_x|^2 + |E_y|^2, |E_x|^2 + |E_y|^2) <--- use delta init.    phi shape: [Nfft-ntaps+1, Nmodes]
        # conv1d_t: t, dtaps, rtap, strides, mode
        td = x.t
        scale = 1e-3*10**(x.power/10)/Nmodes
        y = x.val[t.start - td.start: t.stop - td.stop + x.val.shape[0]] * jnp.exp(-1j*phi * self.gamma * scale * self.dz)
        return x.replace(val=y, t=t)


class TaskMLP(nn.Module):
    features: int=100

    @nn.compact
    def __call__(self, task_info):
        '''
        task_info: [2]
        '''
        return nn.Sequential([nn.Dense(100), nn.relu, nn.Dense(10), nn.relu, nn.Dense(self.features, kernel_init=zeros)])(task_info)



class rotation(nn.Module):
    '''
    rotation a phase per pol state.
    '''
    @nn.compact
    def __call__(self, x: MySignal) -> MySignal:
        '''
            [Nfft, Nmodes] -> [Nfft, Nmodes]
        '''
        Nmodes = x.val.shape[-1]
        theta = self.param('rotation', lambda *_:jnp.zeros(Nmodes, dtype=jnp.float32))
        y = x.val * jnp.exp(1j*theta[None,:])
        return x.repalce(val=y)



class downsamp(nn.Module):
    '''
        down sample MySignal to self.sps.
    '''
    taps:int=32   # kernel shape.
    sps: int=1    # target sps.
    @nn.compact
    def __call__(self, x:MySignal) -> MySignal:
        '''
            [Nfft, Nmodes] -> [Nfft//sps, Nmodes], Nfft//sps=Nsymb
        '''
        assert x.val.shape[0] % x.t.sps == 0, f'x.val.shape[0] must be evenly divided by sps'
        assert x.t.sps % self.sps == 0
        Nmodes = x.val.shape[-1]
        sps = x.t.sps
        rate = sps // self.sps
        y = nn.Conv(features=Nmodes, kernel_size=(self.taps,),strides=(rate,), kernel_init=self.down_kernel, param_dtype=jnp.complex64,dtype=jnp.complex64, padding='valid')(x.val)
        t = conv1d_t(x.t, self.taps, None, sps, 'valid')
        Fs = x.Fs / sps
        return x.replace(val=y, t=t, Fs=Fs)
    

    def down_kernel(self, key, shape, dtype):
        # shape=(taps, dims, dims)
        h = jnp.zeros(shape,dtype=dtype)
        c = (self.taps-1) //2
        for i in range(shape[-1]):
            h = h.at[c,i,i].set(1.0)
        return h
    



class LDBP(nn.Module):
    ''' 
    Learn DBP module.
        steps:int=1
        dtaps:Any=None
        ntaps:Any=None
        L:float=2000e3 # [m]
        freqspace: float=200e9 # [Hz]
        D: float=16.5  # [ps/nm/km]
        Fc: float=299792458/1550E-9 # [Hz] Interested channel center freqs
        F0: float=299792458/1550E-9 # [Hz] WDM center freqs
        gamma: float= 0.0016567 # [/W/m]
    '''
    
    step:int=1                    # DBP steps.
    dtaps:Any=None                 # dispersion kernel size.
    ntaps:Any=None                 # nonlinear kernel size.
    d_init:Callable=delta          # Dispersion kernel initilization function. 'delta', 'gauss', 'zeros'
    n_init:Callable=delta          # Nonlinear kernel initilization function. 'delta', 'gauss', 'zeros'
    meta: str='Filter'             # Nonlinear operator form: 'Filter', 'Fix', 'CNN', 'GRU', 'ConvGRU'
    linear_kernel_type: str='conv' # linear_kernel_type.  'physical' or 'conv'
    linear_share:bool=False        # share linear layer or not.
    nonlinear_share:bool=False     # share nonlinear layer or not.
    optic_params_trainable: bool=False # optic params trainable or not.
    

    nchs:tuple=()               # CNN channels.
    activation:Any=jax.nn.silu  # CNN activation.       
    hidden_dim:int=1            # GRU, hidden dims.
    GRU_depth: int=1            # GRU depth.
    batch_size:int=1000         # GRU train batch size
    
    L:float=2000e3                # [m]
    D: float=16.5                 # [ps/nm/km]
    Fc: float=299792458/1550E-9   # [Hz]  WDM center freqs
    gamma: float= 0.0016567       # [/W/m]
    @nn.compact
    def __call__(self, x:MySignal) -> MySignal:
        '''
            (MySignal, float) -> MySignal
            [Nfft, Nmodes] -> [Nsymb - ol, Nmodes]
        '''
        ## prepare stage
        Nmodes = x.val.shape[-1]
        dz = self.L / self.step  # [m]

        ## DBP stage
        if self.optic_params_trainable:
            D = self.param('DBP D',lambda *_:jnp.array(self.D))
            beta2 = get_beta2(D, self.Fc)/1e3
            beta1 = get_beta1(D, self.Fc, x.Fi)/1e3
            xi = self.param('DBP xi',lambda *_:jnp.ones(self.step))
            gamma = self.param('DBP gamma',lambda *_:jnp.array(self.gamma)) if Nmodes == 1 else self.param('DBP gamma',lambda *_:jnp.array(self.gamma * 8/9))
        else:
            D = self.D
            beta2 = get_beta2(self.D, self.Fc)/1e3
            beta1 = get_beta1(self.D, self.Fc, x.Fi)/1e3
            xi = jnp.ones(self.step)
            gamma = self.gamma if Nmodes == 1 else self.gamma * 8/9

        z = self.L
        Dz = jax.nn.softmax(xi) * self.L

        if self.linear_share: LinOp = Linear(dz=-dz, dtaps=self.dtaps, beta2=beta2, beta1=beta1, kernel_type=self.linear_kernel_type, kernel_init=self.d_init)
        if self.nonlinear_share: NonOp = Nonlinear(dz=Leff(z,-dz), ntaps=self.ntaps,nchs=self.nchs, activation=self.activation,hidden_dim=self.hidden_dim,GRU_depth=self.GRU_depth,gamma=gamma, kernel_init=self.n_init, meta=self.meta, batch_size=self.batch_size)
        
        for i in range(self.step):
            x = LinOp(x) if self.linear_share    else Linear(dz=-Dz[i], dtaps=self.dtaps, beta2=beta2, beta1=beta1, kernel_type=self.linear_kernel_type, kernel_input=self.d_init)(x)
            x = NonOp(x) if self.nonlinear_share else Nonlinear(dz=Leff(z,-Dz[i]), ntaps=self.ntaps,nchs=self.nchs, activation=self.activation,hidden_dim=self.hidden_dim, GRU_depth=self.GRU_depth, gamma=gamma, kernel_init=self.n_init, meta=self.meta,batch_size=self.batch_size)(x)   
            z = z - dz
        return x
    


class DSP(nn.Module):
    model_info: Any
    mode: str='train'

    def setup(self):

        if self.mode == 'train':
            mimo_train = True
        elif self.mode == 'test':
            mimo_train = piecewise_constant([self.model_info['struct_info']['lead_symbols']], [True, False])
        else:
            raise ValueError('invalid mode %s' % self.mode)

        if 'DBP'      in self.model_info['struct_info']['module']: self.DBP = LDBP(name='DBP', **self.model_info['DBP_info'])
        if 'MIMOAF'   in self.model_info['struct_info']['module']: self.MIMOAF = mimoaf(name='MIMOAF',train=mimo_train, **self.model_info['mimo_info'])
        if 'MetaMIMO' in self.model_info['struct_info']['module']: self.MetaMIMO = MetaMIMO(name='MetaMIMO',train=mimo_train, **self.model_info['mimo_info'])
        if 'downsamp' in self.model_info['struct_info']['module']: self.downsamp = downsamp(taps=self.model_info['mimo_info']['taps'])
        if 'BST'      in self.model_info['struct_info']['module']: self.BST = batch_state()

    def __call__(self, x: MySignal, truth: MySignal, update_state:bool) -> MySignal:

        if 'DBP'      in self.model_info['struct_info']['module']: x = self.DBP(x)
        if 'MIMOAF'   in self.model_info['struct_info']['module']: x = self.MIMOAF(x, truth, update_state)
        if 'MetaMIMO' in self.model_info['struct_info']['module']: x = self.MetaMIMO(x, truth, update_state)
        if 'downsamp' in self.model_info['struct_info']['module']: x = self.downsamp(x)
        if 'BST'      in self.model_info['struct_info']['module']: x = self.BST(x)
        return x

    



def init_model(model_info,mode='train', initialization=True, batch_dim=2, update_state=True, data_train=None):
    '''
        Input:
            model_info: dict
            mode: str. 'train' or 'test'
            initialization: bool. True for initialization, False for not.
            batch_dim: bool. True for batch dimension, False for not.
            update_state: bool. True for update state, False for not.
            data_train: dataset for initialization.
        Output:
            if initialization:
                return net, params, state, ol
            else:
                return net
    '''
    # step 1: get information
    steps = model_info['DBP_info']['step']      # [steps]
    dtaps = model_info['DBP_info']['dtaps']     # [samples]
    steps = model_info['DBP_info']['step']      # [steps]
    ntaps = model_info['DBP_info']['ntaps']     # [samples]
    Fs = model_info['private_info']['Fs']        # [Hz]     ----- x.Fs
    Fi = model_info['private_info']['Fi']        # [Hz]     ----- x.Fi
    sps = model_info['private_info']['sps']      # [samples/symbol]   ----- x.t.sps
    Nmodes = model_info['private_info']['Nmodes'] # [modes]  

    # step 2: pretrain dispersion initialization.  (in 'conv' mode only)
    if model_info['DBP_info']['linear_kernel_type'] == 'conv':
        L =     model_info['DBP_info']['L']         # [m]
        D =     model_info['DBP_info']['D']         # [ps/nm/km]
        Fc =    model_info['DBP_info']['Fc']        # [Hz]
        beta2 = get_beta2(D, Fc)/1e3
        beta1 = get_beta1(D, Fc, Fi)/1e3    
        kernel = pretrain_kernel(-L/steps, dtaps, Fs, beta2, beta1, 'time', steps, pre_train_steps=200)
        d_init = lambda rng,Nmodes: jnp.stack([kernel]*Nmodes, axis=-1)
        DBP_info = {'d_init':d_init, **model_info['DBP_info']}
    else:
        DBP_info = model_info['DBP_info']
    settings = freeze({'struct_info':model_info['struct_info'], 'DBP_info':DBP_info, 'mimo_info':model_info['mimo_info'], 'private_info':model_info['private_info']})
    
    # step 3: initialization
    VmapDSP = nn.vmap(DSP, variable_axes={'params':None, 'const':None, 'state':0},     # parameters share for axis=0.
                   split_rngs={'params':False, 'const':False, 'state':True},
                   in_axes=(0,0,None), out_axes=0)
    net = VmapDSP(settings, mode=mode) if batch_dim else DSP(settings, mode=mode)
    init_len = int((dtaps + np.sum(ntaps))*steps) + 2000
    init_len = int(init_len - (init_len % sps))

    # sequence for state_init convergence.
    if data_train == None:
        if batch_dim:
            y = MySignal(jnp.ones([batch_dim, init_len,Nmodes], dtype=jnp.complex64), t=SigTime(0,0,sps), Fs=Fs*jnp.ones(batch_dim), power=0*jnp.ones(batch_dim), Fi=Fi*jnp.ones(batch_dim), Nch=jnp.ones(batch_dim))
            x = MySignal(jnp.ones([batch_dim, init_len//sps,Nmodes], dtype=jnp.complex64), t=SigTime(0,0,1), Fs=Fs//sps*jnp.ones(batch_dim), power=0*jnp.ones(batch_dim), Fi=Fi*jnp.ones(batch_dim), Nch=jnp.ones(batch_dim))
        else:
            y = MySignal(jnp.ones([init_len,Nmodes], dtype=jnp.complex64), t=SigTime(0,0,sps), Fs=Fs, power=0, Fi=Fi, Nch=1)
            x = MySignal(jnp.ones([init_len//sps,Nmodes], dtype=jnp.complex64), t=SigTime(0,0,1), Fs=Fs//sps, power=0, Fi=Fi, Nch=1)
    else:
        if data_train.y.ndim == 2 and batch_dim: raise(ValueError)
        if data_train.y.ndim == 3 and not batch_dim: raise(ValueError)
        y = MySignal(data_train.y[..., 0:init_len,:],          t=SigTime(0,0,sps), Fs=Fs*jnp.ones(1), power=0*jnp.ones(1), Fi=Fi*jnp.ones(1))
        x = MySignal(data_train.x[..., 0:(init_len//sps),:], t=SigTime(0,0,1), Fs=Fs//sps*jnp.ones(1), power=0*jnp.ones(1), Fi=Fi*jnp.ones(1))

    if initialization:
        z, v = net.init_with_output(rd.PRNGKey(0), y, x, update_state)
        ol = z.t.start - z.t.stop
        params = v['params']
        state = v['state']
        return net, params, state, ol
    else:
        return net


def construct_update(net:nn.Module, tx: optax.GradientTransformation, device: str='cpu', loss_type='logMSE') -> Callable:
    '''
        Input: 
            net: flax module.    net.apply: y,x,update_state -> x_preditcted, new_state
            tx: optax optimizer.
            device: 'cpu' or 'gpu'
        Output:
            Callable update function.
            params, state, opt_state,y_data,x_data -> params, state, opt_state
    '''


    def loss_fn(params:dict,state:dict, y:MySignal, x:MySignal, update_state:bool=True)->float:
        '''
            predict x from y.
            y.val: [batch, Nfft, Nmodes]  or [Nfft, Nmodes] 
            x.val: [batch, Nsymb, Nmodes]  or [Nsymb, Nmodes] 
        '''
        v = {'params': params,'state':state}
        x_predict, new_state_dic = net.apply(v, y, x, update_state, mutable='state')  # Input truth as pilotsymb
        new_state = new_state_dic['state']
        start = x_predict.t.start
        stop = x_predict.t.stop
        if loss_type=='logMSE':
            return jnp.log(jnp.mean(jnp.abs(x.val[...,start:stop,:] - x_predict.val)**2)), new_state
        elif loss_type=='MSE':
            return jnp.mean(jnp.abs(x.val[...,start:stop,:] - x_predict.val)**2), new_state
        else:
            raise ValueError
        # return jnp.mean(jnp.abs(x.val[...,start:stop,:] - x_.val)**2 / jnp.abs(x.val[...,start:stop,:])**2), new_state

    @partial(jax.jit, backend=device)
    def update_step(params, state, opt_state, y:MySignal, x:MySignal):
        '''
        params, state, opt_state, y, x -> params, state, opt_state, l
        '''
        (l, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, state, y, x)
        grads = jax.tree_map(lambda x:jnp.conj(x), grads)
        uptdates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, uptdates)
        return params, new_state, opt_state, l
    
    return update_step



## Meta MIMO

from .adaptive_filter import mimoinitializer, mimo, decision, Any, Tuple, Array, QAM, make_schedule
from .core import MySignal, conv1d_t, SigTime
from .activation import csigmoid, ctanh
from .operator import frame
from .layers import LSTMCell



def schedule(i):
    return True



from .MetaOptimizer import MetaAdaGradOpt, MetaAdamOpt, MetaLSTMOpt_A, MetaLSTMOpt_B, MetaRmspropOpt, MetaSGDOpt

class MetaMIMOCell(nn.Module):
    taps: int=32
    train: Any = schedule
    dims: int=2
    grad_max: Tuple[float, float] = (30., 30.)
    const: Array = QAM(16).const()
    
    MetaOpt: nn.Module = MetaAdamOpt(learning_rate_init=1/2**7)
    @nn.compact
    def __call__(self, state, inp):
        # step 1: data, gradient
        u,x = inp
        grads, d,z,e = self.grad(state['theta'], inp, state['iter'])

        # Step 2: update hidden state (Fix learning rate)
        # hidden, updates = state['hidden'], jax.tree_map(lambda x,y: -x*y, grads, self.lr_init)

        # Step 2: update hidden state (Meta learning rate)
        add_in = state['theta']
        hidden, updates = self.MetaOpt(state['hidden'], grads, add_in)

        # Step 3: update parameters
        theta0 = state['theta']
        theta = optax.apply_updates(state['theta'], updates)
        iter = state['iter'] + 1
        return {'iter':iter, 'theta':theta, 'hidden':hidden}, (theta0, grads)

    
    def init_carry(self, dtype=jnp.complex64, mimoinit='zeros'):
        w0 = mimoinitializer(self.taps, self.dims, dtype, mimoinit) 
        f0 = jnp.full((self.dims,), 1., dtype=dtype)
        theta = (w0, f0)
        hidden = self.MetaOpt.init_carry(theta)

        return {'iter': jnp.zeros((),dtype=int), 'theta':theta, 'hidden': hidden}
    

    
    def grad(self, theta, inp, i):
        u,x = inp
        w, f = theta
        v = mimo(w, u)
        k = v * f
        d = jnp.where(self.train(i), x, decision(self.const, k))
        l = jnp.sum(jnp.abs(k - d)**2)

        psi_hat = jnp.abs(f)/f 
        e_w = d * psi_hat - v
        e_f = d - k
        gw = -1. / ((jnp.abs(u)**2).sum() + 1e-9) * e_w[:, None, None] * u.conj().T[None, ...]
        gf = -1. / (jnp.abs(v)**2 + 1e-9) * e_f * v.conj()
        # bound the grads of f and s which are less regulated than w,
        # it may stablize this algo. by experience
        gf = jnp.where(jnp.abs(gf) > self.grad_max[0], gf / jnp.abs(gf) * self.grad_max[0], gf)

        return (gw, gf), d, k, (e_w, e_f)
    
    # def grad(self, theta, inp, i):
    #     u,x = inp
    #     w, f = theta
    #     v = mimo(w, u)
    #     k = v * f / jnp.abs(f)
    #     d = jnp.where(self.train(i), x, decision(self.const, k))
    #     l = jnp.sum(jnp.abs(k - d)**2)

    #     psi_hat = jnp.abs(f)/f 
    #     e_w = d * psi_hat - v
    #     e_f = d - k
        
    #     grads = jax.grad(self.loss_fn)(theta, u, d)
    #     gw = grads[0].conj() / ((jnp.abs(u)**2).sum() + 1e-9)
    #     gf =  grads[1].conj() / (jnp.abs(v)**2 + 1e-9)
    #     gf = jnp.where(jnp.abs(gf) > self.grad_max[0], gf / jnp.abs(gf) * self.grad_max[0], gf)

    #     return (gw, gf), d, k, (e_w, e_f)

    def loss_fn(self, theta, y, d):
        w,f = theta
        return jnp.sum(jnp.abs(mimo(w, y)*f/jnp.abs(f) - d)**2)

    def apply_fn(self, theta, yf):
        ws, fs = theta
        return jax.vmap(mimo)(ws, yf) * fs 


class MetaMIMO(nn.Module):
    taps: int=32
    train: Any = schedule
    grad_max: Tuple[float, float] = (30., 30.)
    const: Array = QAM(16).const()
    MetaOpt: nn.Module = MetaLSTMOpt_B()
    @nn.compact
    def __call__(self, signal:MySignal, truth:MySignal, update_state:bool) -> MySignal:
        x = signal.val
        dims = x.shape[-1]
        sps  = signal.t.sps
        t = conv1d_t(signal.t, self.taps, None, sps, 'valid')
        x = frame(x, self.taps, sps) # [Nsymb, taps, Nmodes]
        if truth is not None: truth = truth.val[t.start: truth.val.shape[-2] + t.stop]          # [Nsymb, Nmodes]

        ScanCell = nn.scan(MetaMIMOCell, variable_broadcast='params',split_rngs={'params': False})(taps=self.taps, train=make_schedule(self.train), dims=dims, grad_max=self.grad_max, const=self.const, MetaOpt=self.MetaOpt)
        state = self.variable('state', 'MetaState', lambda *_:ScanCell.init_carry(),())
        state_new, (af_weights, Delta) = ScanCell(state.value, (x, truth))
        y = ScanCell.apply_fn(af_weights, x) # 所有的 symbol 都过了之后才运用filter
        if update_state: state.value = state_new
        return signal.replace(val=y, t=t, Fs=signal.Fs/sps)