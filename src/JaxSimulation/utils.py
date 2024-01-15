import jax, jax.numpy as jnp, jax.random as random, flax.linen as nn, matplotlib.pyplot as plt
from functools import partial, wraps
from typing import Any, NamedTuple
from flax.core import freeze, unfreeze
import os, sys, time,numpy as np
from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict, unflatten_dict
from scipy import constants as const

class parameters:
    """
    Basic class to be used as a struct of parameters
    """
    pass


def is_same_structure(a,b):
    '''
    Compare the structure of a,b
        a,b: Pytree
        return: True or false
    '''
    _,t1 = jax.tree_util.tree_flatten(a)
    _,t2 = jax.tree_util.tree_flatten(b)
    return t1==t2


def show_tree(tree):
    '''
    Show pytree shape infirmation.
    Input:
        a Pytree.
    Output:
        a pytree has the same structrue with input carry the shape information.
    '''
    return jax.tree_map(lambda x:x.shape, tree)

def c2r(x):
    '''
    [shape] --> [2,shape]
    '''
    if (x.dtype == jnp.complex64) or (x.dtype == jnp.complex128):
        return jnp.array([x.real, x.imag])
    else:
        return jnp.array([x])

def r2c(x):
    '''
    x: [2,shape] --> [shape]
    '''
    if x.shape[0] == 2:
        return x[0] + (1j)*x[1]
    else:
        return x[0]

def tree_c2r(var, key='params'):
    '''
    把 var 中的 params 变成实参数
    '''
    var = unfreeze(var)
    var[key] = jax.tree_map(c2r,var[key])
    return freeze(var)

def tree_r2c(var, key='params'):
    '''
    把 var 中的 params 变成复参数
    '''
    var = unfreeze(var)
    var[key] = jax.tree_map(r2c,var[key])
    return freeze(var)

class realModel(NamedTuple):
    init: Any
    apply: Any
    init_with_output: Any

def realize(model):
    '''
    Make a model pure real.
    '''
    @wraps(model.init)
    def _init(*args, **kwargs):
        var =  model.init( *args, **kwargs)
        return tree_c2r(var)
    
    @wraps(model.apply)
    def _apply(var_real, *args, **kwargs):
        var = tree_r2c(var_real)
        out = model.apply(var, *args, **kwargs)
        return out
    
    @wraps(model.init_with_output)
    def _init_with_output(key, *args, **kwargs):
        z,v = model.init_with_output(key, *args, **kwargs)
        return z, tree_c2r(v)
    
    return realModel(init=_init,apply=_apply, init_with_output=_init_with_output)

def pmap(model):
    '''
    pmap a model.
    '''
    @wraps(model.init)
    def _init(*args, **kwargs):
        var =  model.init( *args, **kwargs)
        return tree_c2r(var)
    
    @wraps(model.apply)
    def _apply(var_real, *args, **kwargs):
        var = tree_r2c(var_real)
        out = model.apply(var, *args, **kwargs)
        return out
    
    @wraps(model.init_with_output)
    def _init_with_output(key, *args, **kwargs):
        z,v = model.init_with_output(key, *args, **kwargs)
        return z, tree_c2r(v)
    
    return realModel(init=_init,apply=_apply, init_with_output=_init_with_output)




def nn_vmap_x(module):
    '''
    将 net vmap 到 Signal第一个分量的axis=-1上, 并且参数不共享. 
    输入为 Array
    '''
    return nn.vmap(module, 
    variable_axes={'params':-1},  # 表示变量'params'会沿着axis=-1复制, 'const'不会复制
    split_rngs={'params':True}, # 表示初始化的种子会split
    in_axes=-1, out_axes=-1) 


def nn_vmap_batch(module):
    # 将 [Nfft,Nmodes] 输入扩展为 [batch, Nfft, Nmodes]
    return nn.vmap(module, 
                   variable_axes={'params':None, 'const':None, 'state':0},     # parameters share for axis=0.
                   split_rngs={'params':False, 'const':False, 'state':False})



def MSE(y,y1):
    return jnp.sum(jnp.abs(y-y1)**2)




def calc_time(f):
    
    @wraps(f)
    def _f(*args, **kwargs):
        t0 = time.time()
        y = f(*args, **kwargs)
        t1 = time.time()
        print(f' {f.__name__} complete, time cost(s):{t1-t0}')
        return y
    return _f



class HiddenPrints:
    '''
    Hidden Print informations.!
    '''
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def make_init(f):
    @wraps(f)
    def _f(key, *args, **kwargs):
        return f(*args, **kwargs)


def show_symb(sig,symb,s=10,figsize=(8,4),title='title'):
    '''
    sig,symb: L2 = 1    [Nsymb, Nmodes]
    '''
    Nmodes = sig.shape[-1]
    symb_set = np.unique(symb)

    fig, ax = plt.subplots(1,2, figsize=figsize)
    fig.suptitle(title)

    for p in range(Nmodes):
        for sym in symb_set:
            
            z = sig[...,p][symb[...,p] == sym]
            ax[p].scatter(z.real, z.imag, s=s)



def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return FrozenDict({k: (map_fn(v) if isinstance(v, FrozenDict) else fn(k, v))
            for k, v in nested_dict.items()})
  return map_fn



def params_label(params):
    params_dict = flatten_dict(params)
    label_dict= {}
    for k in params_dict.keys():
        label_dict[k] = k[0].split('_')[-1]
    return FrozenDict(unflatten_dict(label_dict))




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

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P


def get_dtaps(steps, a:dict, dscale:float=1):
    '''
        calculate the dtaps of dispersion.
    '''
    pi = np.pi
    C = 299792458.         # [m/s]
    lambda_ = C / jnp.mean(a['carrier_frequency'])
    beta2 = a['D'] * lambda_**2 / (2 * np.pi * C)
    return estimate_dtaps(a['distance']/steps, jnp.max(a['samplerate']), beta2, dscale)


def estimate_dtaps(L, samplerate, beta2, dscale):
    '''
        estimate dtaps in time domain.
    Input:
        L: distance.[m]
        samplerate: [Hz]
        beta2: diapersion coeff. [s^2/m]
        dscale: control the ratio.
    Output:
        int.

    '''
    pi = np.pi
    sr = samplerate        # [Hz]
    mintaps = int(np.ceil(2 * pi * L * beta2 * sr**2) * dscale)
    return mintaps - (mintaps % 4) + 5
    

    
def get_beta1(D, Fc, Fi):
    '''
        Input:
            D:[ps/nm/km]    Fc: [Hz]   Fi: [Hz] 
        Output:
            beta1:    [s/km]
    '''
    beta2 = get_beta2(D, Fc)  # [s^2/km]
    beta1 = 2*np.pi * (Fi - Fc)*beta2 # [s/km]
    return beta1

def get_beta2(D, Fc):
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

    

def smooth(data, window_size):
    """
    对一维的数据列表进行滑动平均

    Args:
        data (list): 一维的数据列表
        window_size (int): 窗口大小，即取前后多少个数据点的平均值

    Returns:
        numpy.ndarray: 平滑化后的数据列表
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'valid')