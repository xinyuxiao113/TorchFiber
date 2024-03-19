import jax, numpy as np, jax.numpy as jnp, matplotlib.pyplot as plt
from commplax import equalizer as eq 
from commplax import comm
from commplax.module import core
Array = jax.Array
from src.JaxSimulation.operator import corr_circ


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



def auto_rho(x: Array,y: Array) -> Array:
    '''
        auto-correlation coeff.
    Input:
        x: Array 1. (N,)
        y: Array 2. (N,)
    Output:
        Correlated coefficients of x,y.(N,)
    '''
    N = len(x)
    Ex = jnp.mean(x)
    Ey = jnp.mean(y)
    Vx = jnp.var(x)
    Vy = jnp.var(y)
    return (corr_circ(x,y)/N - Ex*Ey)/jnp.sqrt(Vx*Vy)

def time_recovery(y,x):
    '''
    predict sequence: y
    truth: x
    '''
    i = jnp.argmax(auto_rho(jnp.abs(x), jnp.abs(y)).real)
    return jnp.roll(y, i), i

time_recovery_vmap = jax.vmap(time_recovery, in_axes=(-1, -1), out_axes=-1)


def simple_cpr(sigRx, symbTx, discard=100):
    '''
    sigRx: [N,2] have done!
    '''
    ind = np.arange(discard, sigRx.shape[0] - discard)
    rot = np.mean(symbTx[ind]/sigRx[ind], axis=0)
    sigRx  = rot[None,:] * sigRx

    y = []
    for i in range(sigRx.shape[1]):
        y.append(sigRx[:,i]/np.sqrt(signal_power(sigRx[ind,i])))
    return jnp.stack(y, axis=-1)

def mimo_dsp(data, eval_range=(30000, -20000), metric_fn=comm.qamqot):
    '''
        data: Array. [Nfft, Nmodes]
    '''
    y = []
    x = data.x
    polmux = (data.y.shape[-1]>1)
    y.append(data.y)
    y.append(eq.cdcomp(y[0], data.a['samplerate'], CD=data.a['CD'], polmux=polmux))
    y.append(eq.modulusmimo(y[1], taps=21, lr=2**-14)[0])  # 这一步可能把符合序号映射错
    y.append(time_recovery_vmap(y[2], x)[0])
    y.append(eq.qamfoe(y[3])[0])
    y.append(eq.ekfcpr(y[4])[0])
    y.append(simple_cpr(y[5], x) )
    
    sig_list = {}
    sig_list['Rx'] = core.Signal(y[0],core.SigTime(0,0,2))
    sig_list['CDC'] =  core.Signal(y[1],core.SigTime(0,0,2))
    sig_list['MIMO'] = core.Signal(y[2],core.SigTime(0,0,1))
    sig_list['Time Recovery'] = core.Signal(y[3],core.SigTime(0,0,1))
    sig_list['FOE'] = core.Signal(y[4],core.SigTime(0,0,1))
    sig_list['CPR'] = core.Signal(y[5],core.SigTime(0,0,1))
    sig_list['rotation'] = core.Signal(y[6],core.SigTime(0,0,1))


    z = sig_list['rotation'] 
    metric = metric_fn(z.val,
                        data.x[z.t.start:data.x.shape[0] + z.t.stop],
                        scale=jnp.sqrt(10),
                        eval_range=eval_range)

    return sig_list, metric



def show_symb(sig, symb, name, idx1, idx2, size=10, fig_size=(15,4), time_recovery = True):

    ## constellation
    symb_set = np.unique(symb[:,0])

    sig_ = sig.val[::int(sig.t.sps)][idx1]
    symb_ = symb[sig.t.start//int(sig.t.sps): symb.shape[0] + sig.t.stop//int(sig.t.sps)][idx1]
    if time_recovery:
        sig_ =  time_recovery_vmap(sig_, symb_)[0]
    modes = symb_.shape[1]
    

    fig, ax = plt.subplots(1,4, figsize=fig_size)
    fig.suptitle(name)

    for sym in symb_set:
        for j in range(modes):
            sigj = sig_[:,j]
            symbj = symb_[:,j]
            z = sigj[symbj == sym]

            ax[j].scatter(z.real, z.imag, s=size)

    ## angle error, t的单位是 symbol period
    sig_ = sig.val[::int(sig.t.sps)][idx2]
    symb_ = symb[sig.t.start//int(sig.t.sps): symb.shape[0] + sig.t.stop//int(sig.t.sps)][idx2]
    if time_recovery:
        sig_ =  time_recovery_vmap(sig_, symb_)[0]
    modes = sig_.shape[1]

    for j in range(modes):
        sigj = sig_[:,j]
        symbj = symb_[:,j]
        ax[2+j].plot(np.angle(sigj/symbj))




def show_fig(sig_list, symb, idx1=None,idx2=None,point_size=10, fig_size=(15,4)):
    if id(idx1) == id(None):
        idx1 = np.arange(symb.shape[0]//3, symb.shape[0]//3 + 10000)
    if id(idx2) == id(None):
        idx2 = np.arange(symb.shape[0]//3, symb.shape[0]//3 + 600)
    
    for name,sig  in sig_list:
        show_symb(sig, symb, name, idx1,idx2,size=point_size, fig_size=fig_size)
