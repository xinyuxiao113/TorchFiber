from typing import Callable, Any, Optional
from flax import linen as nn
from flax import struct
import jax.numpy as jnp, jax.random as rd
import jax
from jax import lax
import numpy as np
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,Union)
from flax.linen.initializers import lecun_normal
from .activation import cleaky_relu, csigmoid, ctanh, csilu, complex_F, crelu, complex_sigmoid
from .operator import frame, convolve
from .initializers import zeros, near_zeros, gauss, complex_variance_scaling


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]



class fconv1d(nn.Module):
    features:int
    kernel_size:int
    strides:int=1
    padding:str='valid'
    use_bias:bool=True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    kernel_init:Callable=lecun_normal()
    bias_init:Callable=zeros
    conv_fn: Callable=convolve
    @nn.compact
    def __call__(self, x):
        '''
        x: (N,Ci)
        '''
        Ci = x.shape[-1]
        Co = self.features
        k = np.sum(self.kernel_size)
        h_ = self.param('kernel', self.kernel_init, (k, Ci, Co), self.param_dtype)
        h = jnp.reshape(h_, (h_.shape[0], Ci*Co))
        y = jnp.tile(x, (1,Co))
        zflat = jax.vmap(lambda a, b: self.conv_fn(a, b, mode=self.padding), in_axes=-1, out_axes=-1)(y, h)
        z = zflat.T.reshape((Co, Ci, -1)).sum(axis=1).T # [N,Co]
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (Co,), self.param_dtype) # [Co]
            sps = int(np.sum(self.strides))
            return z[::sps,:] + bias[None,...]
        else:
            return z[::self.strides,:]

class cnn_block(nn.Module):
    kernel_shapes: tuple=(3,5,3,5)  # (3,3,3)
    channels: tuple=(4,8,4,2)       # (2,2,2)
    padding: str='valid'
    dtype:Any = jnp.complex64
    param_dtype:Any = jnp.complex64
    activation:Any= csilu
    n_init:Callable=gauss
    conv1d:Any=nn.Conv
    '''
    Input:
    [B, L, 2]  --> [B,L,4] --> [B,L,8] --> []
    '''

    @nn.compact
    def __call__(self,inputs):
        assert len(self.kernel_shapes) == len(self.channels)
        Conv = partial(self.conv1d, strides=(1,), param_dtype=self.param_dtype, dtype=self.dtype, padding=self.padding, kernel_init=self.n_init)
        x = inputs
        for k in range(len(self.kernel_shapes) - 1):
          x = Conv(features=self.channels[k], kernel_size=(self.kernel_shapes[k],))(x)
          x = self.activation(x)
        x = Conv(features=self.channels[-1], kernel_size=(self.kernel_shapes[-1],))(x)
        x = self.activation(x)
        return x


class LRNNCell(nn.Module):
    @nn.compact
    def __call__(self,carry, x):
        '''
        carry: [Nmodes],  x: [Nmodes]
        '''
        eta = self.param('rnn eta', lambda *_:jnp.array(0.1))
        lr = jax.nn.tanh(eta)

        carry_ = carry * lr + x * (1-lr)
        x_ = carry_
        return carry_, x_

class GRU(nn.Module):
    param_dtype:Any=jnp.float32
    dtype:Any=jnp.float32
    kernel_init: Any=near_zeros
    bias_init:Any=near_zeros
    gate_fn:Any=csigmoid
    activation_fn:Any=ctanh

    @nn.compact
    def __call__(self, c, xs):
        NN = nn.scan(nn.GRUCell,
                      variable_broadcast="params",
                      split_rngs={"params": False},
                      in_axes=0,   # along axis=1 scan
                      out_axes=0)  # output on axis=1
        ## 这里可以修改 GRU的init函数
        
        return NN(param_dtype=self.param_dtype, dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                gate_fn=self.gate_fn,
                activation_fn=self.activation_fn)(c, xs)
    



class LRNN_net(nn.Module):
    depth:int=1
    dims: tuple=(2,)

    @nn.compact
    def __call__(self,c,xs):
        NN = nn.scan(LRNNCell, variable_broadcast='params', split_rngs={'params':False}, in_axes=0, out_axes=0)
        return NN()(c,xs)


class GRU_net(nn.Module):
    depth:int=1
    dims: tuple=(2,)
    param_dtype:Any=jnp.float32
    dtype:Any=jnp.float32
    kernel_init: Any=near_zeros
    bias_init:Any=near_zeros
    gate_fn:Any=csigmoid
    activation_fn:Any=ctanh
    
    @nn.compact
    def __call__(self,c,xs):
        for i in range(self.depth):
            c[i],xs = GRU(param_dtype=self.param_dtype, dtype=self.dtype, kernel_init=self.kernel_init,
            bias_init=self.bias_init, gate_fn=self.gate_fn, activation_fn=self.activation_fn)(c[i],xs)
        return c,xs
    
    def carry_init(self):
        assert len(self.dims) == self.depth
        c = []
        for i in range(self.depth):
            c.append(jnp.zeros(self.dims[i]))
        return c


LRNN = GRU_net


class fun1d(nn.Module):
    dims: int=1
    m:int=10
    
    @nn.compact
    def __call__(self,x):
        '''
            x: (*,1)
        '''
        y = nn.Dense(self.m)(x)
        y = nn.relu(y)
        y = nn.Dense(80)(y)
        y = nn.relu(y)
        y = nn.Dense(self.dims, use_bias=True, kernel_init=zeros)(y)
        return y
    
    
    

    

from typing import Any
class LSTMCell(nn.Module):
    dtype: Any=jnp.float32
    param_dtype: Any=jnp.float32
    depth:int = 1
    hidden_dims: int = 5
    gate_fn: Callable = complex_sigmoid
    activation_fn: Callable = ctanh
    @nn.compact
    def __call__(self,c,x):
        c_new = []
        for i in range(self.depth):
            ci,x = nn.OptimizedLSTMCell(dtype=self.dtype, param_dtype=self.param_dtype, gate_fn=self.gate_fn, activation_fn=self.activation_fn)(c[i],x)
            c_new.append(ci)
        return c_new, x

    @staticmethod
    def initialize_carry(key, depth, hidden_dims):
        keys = rd.split(key, depth)
        return [nn.OptimizedLSTMCell.initialize_carry(keys[i], (), hidden_dims) for i in range(depth)]


class NLayerLSTM(nn.Module):
    hidden_dims: List[int]
    dtype: jnp.dtype = jnp.complex64 # type: ignore
    param_dtype: jnp.dtype = jnp.complex64  # type: ignore
    gate_fn: Callable = complex_sigmoid
    activation_fn: Callable = ctanh

    def setup(self):
        self.lstm_layers = [nn.LSTMCell(name=f'lstm_{i}', gate_fn=self.gate_fn, activation_fn=self.activation_fn, dtype=self.dtype, param_dtype=self.param_dtype) for i, _ in enumerate(self.hidden_dims)]  # type: ignore

    def __call__(self, states, inputs):
        new_states = []

        layer_output = inputs
        for lstm, state in zip(self.lstm_layers, states):
            new_state, layer_output = lstm(state, layer_output)
            new_states.append(new_state)

        return new_states, layer_output

    @staticmethod
    def initialize_carry(hidden_dims,batch_size=None, dtype=jnp.float32):
        if batch_size == None:
            return [(jnp.zeros((hidden_dim,),dtype=dtype), jnp.zeros((hidden_dim,), dtype=dtype)) for hidden_dim in hidden_dims]
        else:
            return [(jnp.zeros((batch_size, hidden_dim),dtype=dtype), jnp.zeros((batch_size, hidden_dim), dtype=dtype)) for hidden_dim in hidden_dims]




class MLP(nn.Module):
    features: int=5
    depth: int = 1
    hidden_dims: int = 5
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x):

        ## MetaMIMO
        for i in range(self.depth - 1):
            x = nn.Dense(features=self.hidden_dims, dtype=self.dtype, param_dtype=self.param_dtype)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.features, dtype=self.dtype, param_dtype=self.param_dtype)(x)
        return x