import jax, optax, jax.random as rd, jax.numpy as jnp, flax.linen as nn, numpy as np, matplotlib.pyplot as plt
from .receiver import BER, SER, DataInput
from .core import MySignal, wrap_signal
from .adaptive_filter import piecewise_constant
from .dsp import DBP, CDC, BPS, mimoaf, TimeDBP, downsamp
from .utils import get_dtaps


def mask_signal(x:MySignal, lead_symbols:int) -> MySignal:
    # x.val.shape  [N,Nomdes]
    return x.replace(val = x.val * (jnp.arange(x.val.shape[-2]) < lead_symbols)[:,None]) # type: ignore

def CDCBPS(data_val, lead_symbols=2000, N: int=61, B: int=121):
    '''
        CDC + BPS
    Input: 
        dataset.
        lead_symbols.

    Output:
        metric, recovered symbols.
    '''
    y_val, x_val = wrap_signal(data_val)

    # Step 1: CDC
    y_cdc = CDC(y_val, data_val.a['distance'])

    # Step 2: down sample.
    y_down,_ = downsamp(taps=2).init_with_output(rd.PRNGKey(0), y_cdc)

    # Step 3: BPS
    pilotInd = np.arange(lead_symbols, dtype=int)
    predict = BPS(y_down, x_val, pilotInd=pilotInd, N=N, B=B)

    return predict


def DBPBPS(data_val, steps=8, lead_symbols=2000, N: int=61, B: int=121, order=1):
    '''
        DBP + BPS
    Input: 
        dataset, steps
    Output:
        metric, recovered symbols.
    '''
    y_val, x_val = wrap_signal(data_val)

    # Step 1: DBP
    p = data_val.a['lpdbm']
    dz = data_val.a['distance'] / steps
    y_dbp = DBP(y_val, data_val.a['distance'], dz=dz, power_dbm=p, order=order)

    # Step 2: down sample.
    y_down,_ = downsamp(taps=2).init_with_output(rd.PRNGKey(0), y_dbp)

    # Step 3: BPS
    pilotInd = np.arange(lead_symbols, dtype=int)
    predict = BPS(y_down, x_val, pilotInd=pilotInd, N=N, B=B)

    return predict


def CDCMIMO(data_val,mimotaps=32, lead_symbols=20000):
    '''
        CDC + MIMO
    Input: 
        dataset, mimotaps, lead_symbols.
    Output:
        metric, recovered symbols.
    '''
    mimo_train = piecewise_constant([lead_symbols], [True, False])
    y_val, x_val = wrap_signal(data_val)

    # Step 1: CDC
    y_cdc = CDC(y_val, data_val.a['distance']) 

    # Step 2: MIMO
    x_mask = mask_signal(x_val, lead_symbols)
    MIMOAF = mimoaf(taps=mimotaps, name='MIMOAF',train=mimo_train)  # type: ignore
    predict,_ = MIMOAF.init_with_output(rd.PRNGKey(0), y_cdc, x_mask, True)
    # var= MIMOAF.init(rd.PRNGKey(0), y_cdc, x_mask)                # convergence.
    # z,state = MIMOAF.apply(var, y_cdc, x_mask, mutable='state')   # track.
    
    return predict

def DBPMIMO(data_val, steps=8, mimotaps=32, lead_symbols=20000, order=1):
    '''
        CDC + MIMO
    Input: 
        dataset, mimotaps, lead_symbols.
    Output:
        metric, recovered symbols.
    '''
    mimo_train = piecewise_constant([lead_symbols], [True, False])
    y_val, x_val = wrap_signal(data_val)

    # DBP
    dz = data_val.a['distance'] / steps
    y_dbp = DBP(y_val, data_val.a['distance'], dz=dz, power_dbm=data_val.a['lpdbm'], order=order)

    # Step 2: MIMO
    x_mask = mask_signal(x_val, lead_symbols)
    MIMOAF = mimoaf(taps=mimotaps, name='MIMOAF',train=mimo_train) # type: ignore
    predict,_ = MIMOAF.init_with_output(rd.PRNGKey(0), y_dbp, x_mask, True)
    # var= MIMOAF.init(rd.PRNGKey(0), signal, x_mask)                # convergence.
    # z,state = MIMOAF.apply(var, signal, x_mask, mutable='state')   # track.
    return predict


def DBP_transform(data_val, steps=200, order=1):
    y_val, x_val = wrap_signal(data_val)

    # DBP
    dz = data_val.a['distance'] / steps
    y_dbp = DBP(y_val, data_val.a['distance'], dz=dz, power_dbm=data_val.a['lpdbm'], order=order)
    y = jax.device_get(y_dbp.val)
    return DataInput(y, data_val.x, data_val.w0, data_val.a)


def CDC_transform(data_val):
    y_val, x_val = wrap_signal(data_val)

    # CDC
    y_cdc = CDC(y_val, data_val.a['distance']) 
    y = jax.device_get(y_cdc.val)
    return DataInput(y, data_val.x, data_val.w0, data_val.a)



def TDBPMIMO(data_val, steps=8, mimotaps=32, lead_symbols=20000, dscale=1.2, p_sub = 0):
    '''
        TimeDBP + MIMO
    Input: 
        dataset, mimotaps, lead_symbols.
        p_sub: 降低信号旋转的功率使得性能逼近CDC.
    Output:
        metric, recovered symbols.  
    '''
    mimo_train = piecewise_constant([lead_symbols], [True, False])
    y_val, x_val = wrap_signal(data_val)

    # Step 1: TDBP
    p = data_val.a['lpdbm'] - p_sub
    dz = data_val.a['distance'] / steps
    y_tdbp = TimeDBP(y_val, data_val.a['distance'], dz=dz, power_dbm=p, dscale=dscale)

    # Step 2: MIMO
    x_mask = mask_signal(x_val, lead_symbols)
    MIMOAF = mimoaf(taps=mimotaps, name='MIMOAF',train=mimo_train) # type: ignore
    predict,_ = MIMOAF.init_with_output(rd.PRNGKey(0), y_tdbp, x_mask, True)
    # var= MIMOAF.init(rd.PRNGKey(0), signal, x_mask)                # convergence.
    # z,state = MIMOAF.apply(var, signal, x_mask, mutable='state')   # track.
    return predict