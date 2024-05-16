# type: ignore
import jax, optax, jaxopt
import jax.numpy as jnp, jax.random as rd, numpy as np
from flax import linen as nn
from typing import List
from .activation import ctanh, csigmoid, crelu,complex_sigmoid
from .initializers import complex_variance_scaling
from .adaptive_filter import decision
from .transmitter import QAM
from typing import Callable, Any
from .layers import NLayerLSTM, MLP, NLayerGRU

# concat all items in a pytree       pytree  -> (*,1)
def flat_pytree(pytree):
    leaves, tree = jax.tree_util.tree_flatten(pytree)
    shapes = [leaf.shape for leaf in leaves]
    return jnp.concatenate([jnp.reshape(l, (-1,1)) for l in leaves]), tree, shapes  # tree, [N, 1]

def unflat_pytree(vector, tree, shapes):
    starts = np.cumsum([0] + [np.prod(s) for s in shapes])[:-1]
    leaves = [vector[starts[i]:starts[i]+np.prod(s)].reshape(s) for i, s in enumerate(shapes)]
    return jax.tree_util.tree_unflatten(tree, leaves)

def pre_transform(x):
    mag = jnp.log1p(jnp.abs(x))
    phase = jnp.exp(1.0j * jnp.angle(x))
    return mag * phase

def post_transform(x):
    return x[:,0]


class MetaLSTMOpt_A(nn.Module):
    dtype: jnp.dtype = jnp.complex64   # type: ignore
    hidden_dim: int = 16
    depth: int = 2
    learning_rate_init: float = 1e-3

    def setup(self):
        # control lr in a interval !!
        self.lr = self.param('learning_rate', lambda *_: jnp.array(self.learning_rate_init, dtype=jnp.float32))
        self.RNN = NLayerLSTM(hidden_dims=[self.hidden_dim,]*self.depth,dtype=self.dtype, param_dtype=self.dtype)
        self.linear_in = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype), crelu])
        self.linear_out = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype), 
                                         crelu,
                                         nn.Dense(features=1, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype)])

    def __call__(self, opt_state, grads, params):
        # step 0: choose info to embed
        I0, tree, shapes = flat_pytree(grads)                # (N,1), N = number of parameters.
        add_info, _, s1 = flat_pytree(params)               # (N,1), N = number of parameters.  assert shapes == s1
        # add_info = jnp.stack(add_in * I.shape[0], axis=0) # (N,Nmodes)
        I = jnp.concatenate([I0, add_info], axis=-1)         # (N, Nmodes+1)
        I = pre_transform(I)                                # (N,1)
        I = self.linear_in(I)                               # (N, hidden_dim)
        opt_state, output = self.RNN(opt_state, I)          # hidden: [(N, hidden_dim)x2]xdepth  output: (N, hidden_dim)
        grads = (self.linear_out(output) + I0) * self.lr     # (N, 1)
        grads = unflat_pytree(grads, tree, shapes)          # (taps, Nmodes, Nmodes)
        return opt_state, grads
    

    def init_carry(self, params):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        hidden = [(jnp.zeros((N, self.hidden_dim), dtype=self.dtype),)*2] * self.depth
        return hidden
    



class MetaLSTMOpt_B(nn.Module):
    dtype: jnp.dtype = jnp.complex64
    lstm_depth: int=2
    mlp_depth: int=2
    lstm_width: int=16 
    mlp_width: int=32
    step_max: float = 1e-1


    def setup(self):
        # control lr in a interval !!
        self.LSTM = NLayerLSTM(dtype=jnp.complex64, param_dtype=jnp.complex64, hidden_dims=[self.lstm_width]*self.lstm_depth)      # state, x -> state, y
        self.linear_in = nn.Sequential([nn.Dense(features=self.lstm_width, dtype=jnp.complex64, param_dtype=jnp.complex64), crelu])
        self.linear_out = MLP(features=1, depth=self.mlp_depth, hidden_dims=self.mlp_width, dtype=jnp.complex64, param_dtype=jnp.complex64)

    def __call__(self, opt_state, grads, params):
        # step 0: choose info to embed
        I0, tree, shapes = flat_pytree(grads)               # (N,1), N = number of parameters.
        add_info, _, s1 = flat_pytree(params)               # (N,1), N = number of parameters.  assert shapes == s1
        # add_info = jnp.stack(add_in * I.shape[0], axis=0) # (N, 2)
        I = jnp.concatenate([I0, add_info], axis=-1)        # (N, 2)  complex
        I = pre_transform(I)                                # (N, 2)  complex
        I = self.linear_in(I)                               # (N, hidden_dim)
        opt_state, output = self.LSTM(opt_state, I)         # opt_state: [(N, hidden_dim)x2]xdepth  output: (N, hidden_dim)
        lr = self.linear_out(output) 
        lr = - complex_sigmoid(lr) * self.step_max          # (N,)            
        grads = I0 * lr[:,None]                             # (N, 1)
        grads = unflat_pytree(grads, tree, shapes)          # (taps, Nmodes, Nmodes)
        return opt_state, grads
    

    def init_carry(self, params):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        hidden = [(jnp.zeros((self.lstm_width,), dtype=self.dtype),)*2] * self.lstm_depth
        return hidden


class MetaAdamOpt(nn.Module):
    learning_rate_init: float=1e-3
    b1_init: float=0.9
    b2_init: float=0.999
    eps_init: float=1e-8

    def setup(self):
        self.lr = self.param('learning_rate', lambda *_: jnp.array(self.learning_rate_init, dtype=jnp.float32))
        self.b1 = self.param('b1', lambda *_: jnp.array(self.b1_init, dtype=jnp.float32))
        self.b2 = self.param('b2', lambda *_: jnp.array(self.b2_init, dtype=jnp.float32))
        self.eps = self.param('eps', lambda *_: jnp.array(self.eps_init, dtype=jnp.float32))

    def __call__(self, opt_state, grads, params):
        tx = optax.contrib.split_real_and_imaginary(optax.adam(learning_rate=self.lr, b1=self.b1, b2=self.b2, eps=self.eps))
        uptdates, opt_state = tx.update(grads, opt_state, params)
        return opt_state, uptdates
    
    def init_carry(self, params):
        tx = optax.contrib.split_real_and_imaginary(optax.adam(learning_rate=self.learning_rate_init, b1=self.b1_init, b2=self.b2_init, eps=self.eps_init))
        return tx.init(params)
    

class MetaAdaGradOpt(nn.Module):
    learning_rate_init: float=1e-3
    initial_accumulator_value_init: float=0.1

    def setup(self):
        self.lr = self.param('learning_rate', lambda *_: jnp.array(self.learning_rate_init, dtype=jnp.float32))
        self.initial_accumulator_value = self.param('initial_accumulator_value', lambda *_: jnp.array(self.initial_accumulator_value_init, dtype=jnp.float32))

    def __call__(self, opt_state, grads, params):
        tx = optax.contrib.split_real_and_imaginary(optax.adagrad(learning_rate=self.lr, initial_accumulator_value=self.initial_accumulator_value))
        uptdates, opt_state = tx.update(grads, opt_state, params)
        return opt_state, uptdates
    
    def init_carry(self, params):
        tx = optax.contrib.split_real_and_imaginary(optax.adagrad(learning_rate=self.learning_rate_init, initial_accumulator_value=self.initial_accumulator_value_init))
        return tx.init(params)
    


class MetaRmspropOpt(nn.Module):
    learning_rate_init: float=1e-3
    decay_init: float=0.9
    momentum_init: float=0.9
    nesterov: bool=True

    def setup(self):
        self.lr = self.param('learning_rate', lambda *_: jnp.array(self.learning_rate_init, dtype=jnp.float32))
        self.decay = self.param('decay', lambda *_: jnp.array(self.decay_init, dtype=jnp.float32))
        self.momentum = self.param('momentum', lambda *_: jnp.array(self.momentum_init, dtype=jnp.float32))
    
    def __call__(self, opt_state, grads, params):
        tx = optax.contrib.split_real_and_imaginary(optax.rmsprop(learning_rate=self.lr, decay=self.decay, momentum=self.momentum))
        uptdates, opt_state = tx.update(grads, opt_state, params)
        return opt_state, uptdates
    
    def init_carry(self, params):
        tx = optax.contrib.split_real_and_imaginary(optax.rmsprop(learning_rate=self.learning_rate_init, decay=self.decay_init, momentum=self.momentum_init))
        return tx.init(params)
    


class MetaSGDOpt(nn.Module):
    learning_rate_init: float=1e-3
    momentum_init: float=0.9
    nesterov: bool=True
    def setup(self):
        self.lr = self.param('learning_rate', lambda *_: jnp.array(self.learning_rate_init, dtype=jnp.float32))
        self.momentum = self.param('momentum', lambda *_: jnp.array(self.momentum_init, dtype=jnp.float32))
    
    def __call__(self, opt_state, grads, params):
        tx = optax.contrib.split_real_and_imaginary(optax.sgd(learning_rate=self.lr, momentum=self.momentum, nesterov=self.nesterov))
        uptdates, opt_state = tx.update(grads, opt_state, params)
        return opt_state, uptdates

    def init_carry(self, params):
        tx = optax.contrib.split_real_and_imaginary(optax.sgd(learning_rate=self.learning_rate_init, momentum=self.momentum_init, nesterov=self.nesterov))
        return tx.init(params)
    


class MetaNone(nn.Module):
    lr_init: tuple = (1/2**6, 1/2**7)

    @nn.compact
    def __call__(self, opt_state, grads, params):
        return opt_state, jax.tree_map(lambda x,y: -x*y, grads, self.lr_init)

    def init_carry(self, params):
        return jnp.zeros(1)
    

class MetaLr(nn.Module):
    lr_init: tuple = (1/2**6, 1/2**7)

    @nn.compact
    def __call__(self, opt_state, grads, params):
        return opt_state, jax.tree_map(lambda x,y: -x*y, grads, self.lr_init)

    def init_carry(self, params):
        return jnp.zeros(1)


class MetaGRUOpt(nn.Module):
    dtype: jnp.dtype = jnp.complex64   # type: ignore
    hidden_dim: int = 16
    depth: int = 2
    learning_rate_init: float = 1e-3
    step_max: float=5e-2

    def setup(self):
        self.RNN = NLayerGRU(hidden_dims=[self.hidden_dim,]*self.depth,dtype=self.dtype, param_dtype=self.dtype)
        self.linear_in = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype), crelu])
        self.linear_out = nn.Sequential([nn.Dense(features=1, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype)])
        # self.linear_out = nn.Sequential([nn.Dense(features=self.hidden_dim, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype), 
        #                                  crelu,
        #                                  nn.Dense(features=1, kernel_init=complex_variance_scaling, dtype=self.dtype, param_dtype=self.dtype)])

    def __call__(self, opt_state, grads, params):
        # step 0: choose info to embed
        I0, tree, shapes = flat_pytree(grads)                # (N,1), N = number of parameters.
        add_info, _, s1 = flat_pytree(params)               # (N,1), N = number of parameters.  assert shapes == s1
        # add_info = jnp.stack(add_in * I.shape[0], axis=0) # (N,Nmodes)
        I = jnp.concatenate([I0, add_info], axis=-1)         # (N, Nmodes+1)
        I = pre_transform(I)                                # (N,1)
        I = self.linear_in(I)                               # (N, hidden_dim)
        opt_state, output = self.RNN(opt_state, I)          # hidden: [(N, hidden_dim) ]xdepth  output: (N, hidden_dim)
        lr =  - complex_sigmoid(self.linear_out(output)) * self.step_max   # (N, 1)
        grads = lr * I0                                     # (N, 1)
        grads = unflat_pytree(grads, tree, shapes)          # (taps, Nmodes, Nmodes)
        return opt_state, grads
    

    def init_carry(self, params):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        hidden = [jnp.zeros((N, self.hidden_dim), dtype=self.dtype)] * self.depth
        return hidden
    