import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import wraps

def complex_F(f):
    '''
    split activation function.
    https://arxiv.org/pdf/1802.08026.pdf
    Input:
        f: R -> R
    Output:
        f_: C -> C or R->R
    '''
    @wraps(f)
    def _f(x):
        if (x.dtype == jnp.float32) or (x.dtype == jnp.float64):
            y = f(x)
        elif (x.dtype == jnp.complex64) or (x.dtype == jnp.complex128):
            y = f(x.real) + f(x.imag)*(1j)
        else:
            raise(ValueError)
        return y
    return _f

crelu = complex_F(jax.nn.relu)
cleaky_relu = complex_F(jax.nn.leaky_relu)
ctanh = complex_F(jax.nn.tanh)
csigmoid = complex_F(jax.nn.sigmoid)
csilu = complex_F(jax.nn.silu)

def cid(x):
    return x


class modeReLU(nn.Module):

    @nn.compact
    def __call__(self, x):
        b = self.param('b', lambda *_:-jnp.array(0.01))
        return jax.nn.relu(jnp.abs(x)+b)*x/jnp.abs(x)

relu = jax.nn.relu

def zReLU(z):
    return (z.real > 0)*(z.imag > 0) * z


def LR(x):
    return nn.relu(x)**2/(x**2 + 1)


def complex_sigmoid(x):
    return jax.nn.sigmoid(x.real + x.imag)