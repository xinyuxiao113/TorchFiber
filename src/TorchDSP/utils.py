import torch, numpy as np, torch.nn.functional as F, jax, time
import torch.fft
from functools import partial, wraps
from commpy.modulation import QAMModem

QAM16 = QAMModem(16).constellation / np.sqrt(QAMModem(16).Es)
constellation = torch.tensor(QAM16, dtype=torch.complex64)
def decision(const: torch.Tensor, v: torch.Tensor, stopgrad=True):
    '''
    simple symbol decision based on Euclidean distance
    Input:
        const: [Nconst]   v: [batch, Nmodes]    
    Output:
        d: [batch, Nmodes]
    '''

    d = const[torch.argmin(torch.abs(const[:, None, None] - v[None, ...]), dim=0)]
    return d 


def calc_time(f):
    '''
    Calculation the run time of a function.
    '''
    @wraps(f)
    def _f(*args, **kwargs):
        t0 = time.time()
        y = f(*args, **kwargs)
        t1 = time.time()
        print(f' {f.__name__} complete, time cost(s):{t1-t0}')
        return y
    return _f


def to_device(obj, device):
    '''
    Recursively move tensors to the given device.
    '''
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    else:
        return obj
    

def detach_tree(obj):
    '''
    Recursively detach tensors from the computation graph.
    '''
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: detach_tree(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_tree(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_tree(v) for v in obj)
    else:
        return obj
    

def flat_pytree(pytree):
    leaves, tree = jax.tree_util.tree_flatten(pytree)
    shapes = [leaf.shape for leaf in leaves]
    return torch.concatenate([torch.reshape(l, (-1,1)) for l in leaves]), tree, shapes  # tree, [N, 1]

def unflat_pytree(vector, tree, shapes):
    starts = np.cumsum([0] + [np.prod(s) for s in shapes])[:-1]
    leaves = [vector[starts[i]:starts[i]+np.prod(s)].reshape(s) for i, s in enumerate(shapes)]
    return jax.tree_util.tree_unflatten(tree, leaves)


def tree_map(fn, *trees):
    if len(trees) == 1:
        tree = trees[0]
        if isinstance(tree, (list, tuple)):
            return type(tree)([tree_map(fn, child) for child in tree])
        elif isinstance(tree, dict):
            return {key: tree_map(fn, value) for key, value in tree.items()}
        else:
            return fn(tree)
    else:
        if all(isinstance(tree, (list, tuple)) for tree in trees):
            return type(trees[0])([tree_map(fn, *children) for children in zip(*trees)])
        elif all(isinstance(tree, dict) for tree in trees):
            return {key: tree_map(fn, *[tree[key] for tree in trees]) for key in trees[0]}
        else:
            return fn(*trees)


def pre_transform(x):
    mag = torch.log1p(torch.abs(x))
    phase = torch.exp(1.0j * torch.angle(x))
    return mag * phase


def VmapConv1d(input, filter, stride):
    '''
        [batch, L] x [batch, k] -> [batch, L - k +1]

        output[i,n] = sum_{j} input[i, n-j] * filter[i, j]

        Valid convolution along axis=1, vecterize in batch dimension axis=0.
        Input:
            input: [batch, L]
            filter: [batch, k]
            stride: int
        Output:
            [batch, L - k +1]
    '''
    filter = filter[:,None,:]   # [batch, 1, k]
    return F.conv1d(input, filter, stride=stride, groups=input.shape[0])


def Dconv(input, filter, stride):
    '''
        [batch, L, Nmodes] x [batch, k] -> [batch, L - k +1, Nmodes]

        output[i,n,m] = sum_{j} input[i, k-1+n-j, m] * filter[i, j]

        Dispersion convolution.
        Input:
            input: [batch, L, Nmodes]
            filter: [batch, k]
            stride: int
        Output:
            [batch, L - k +1, Nmodes]
    '''
    return torch.vmap(VmapConv1d, in_dims=(-1, None, None), out_dims=-1)(input, torch.flip(filter, dims=(-1,)), stride)


def Conv1d(input, filter, stride):
    '''
        [Ci, L] x [Co, Ci, ntaps] -> [Co, L - k +1]    or     [batch, Ci, L] x [Co, Ci, ntaps] -> [batch, Co, L - k +1]

        output[b, n, l] = sum_{m, j} input[b, m, l + j] * filter[n, m, j]

        Multi-channel convolution.
        Input:  
            input:  [Ci, L] or [batch, Ci, L]
            filter: [Co, Ci, ntaps]
            stride: int
        Output:
            [Co, L] or [batch, Co, L - k +1]
    '''
    return F.conv1d(input, filter, stride=stride)


def Nconv(input, filter, stride):
    '''
        [batch, L, Ci] x [batch, Co, Ci, ntaps] -> [batch, L - k +1, Co]

        Nonlinear operator convolution.
        Input:
            input:  [batch, L, Ci]
            filter: [batch, Co, Ci, ntaps]
            stride: int
        Output:
            [batch, L - k +1, Co]
    '''
    return torch.vmap(Conv1d, in_dims=(0,0,None), out_dims=0)(input.permute([0,2,1]), filter, stride).permute([0,2,1])



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
    result = torch.fft.ifft(torch.fft.fft(signal, dim=dim) * torch.fft.fft(ker), dim=dim)

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


