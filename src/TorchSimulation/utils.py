import torch, numpy as np, time, matplotlib.pyplot as plt
from functools import wraps, partial
import torch.fft


def calc_time(f):
    
    @wraps(f)
    def _f(*args, **kwargs):
        t0 = time.time()
        y = f(*args, **kwargs)
        t1 = time.time()
        print(f' {f.__name__} complete, time cost(s):{t1-t0}')
        return y
    return _f


# frame, circsum, conv_circ, 

def frame(x: torch.Tensor, flen: int, fstep: int, fnum: int=-1) -> torch.Tensor:
    '''
        generate circular frame from Array x.
    Input:
        x: Arrays about to be framed with shape (B, *dims)
        flen: frame length.
        fstep: step size of frame.
        fnum: steps which frame moved. If fnum==None, then fnum --> 1 + (N - flen) // fstep
    Output:
        A extend array with shape (fnum, flen, *dims)
    '''
    N = x.shape[0]

    if fnum == -1:
        fnum = 1 + (N - flen) // fstep
    
    ind = (np.arange(flen)[None,:] + fstep * np.arange(fnum)[:,None]) % N
    return x[ind,...]


def circsum(a: torch.Tensor, N:int) -> torch.Tensor:
    '''
        Transform a 1D array a to a N length array.
        Input:
            a: 1D Array.
            N: a integer.
        Output:
            d: 1D array with length N.
        
        d[k] = sum_{i=0}^{+infty} a[k+i*N]
    '''
    b = frame(a, N, N)
    t = b.shape[0]*N
    c = a[t::]
    d = torch.sum(b,dim=0)
    d[0: len(c)] = d[0:len(c)] + c
    return d

def conv_circ(signal: torch.Tensor, ker: torch.Tensor, dim=0) -> torch.Tensor:
    '''
    N-size circular convolution.

    Input:
        signal: real Nd tensor with shape (N,)
        ker: real 1D tensor with shape (N,).
    Output:
        signal conv_N ker.
    '''
    ker_shape = [1] * signal.dim()
    ker_shape[dim] = -1
    ker = ker.reshape(ker_shape)
    result = torch.fft.ifft(torch.fft.fft(signal, dim=dim) * torch.fft.fft(ker, dim=dim), dim=dim)

    if not signal.is_complex() and not ker.is_complex():
        result = result.real

    return result


def circFilter(h: torch.Tensor, x: torch.Tensor, dim=0) -> torch.Tensor:
    ''' 
        1D Circular convolution. fft version.
        h: 1D kernel.     x: Nd signal
    '''
    k = len(h) // 2
    h_ = circsum(h, x.shape[dim])
    h_ = torch.roll(h_, -k)
    return conv_circ(x, h_, dim=dim)

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