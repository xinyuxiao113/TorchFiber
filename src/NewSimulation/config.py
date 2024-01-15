import jax, jax.numpy as jnp, numpy as np
from src.JaxSimulation.operator import circFilter
from src.JaxSimulation.transmitter import simpleWDMTx, upsample
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

'''
NLSE
u_z = -i/2u_tt -i gamma|u|^2u f(z)
or 
u_z = -i/2u_tt +i gamma|u|^2u f(z)
'''
# Define t and z grids

# parameters and attenuation function
Ld = 40    # [km] dispersion length
Lnl = 400  # [km] nonlinear length
La = 21.73 # [km] attenuation length
Ls = 80    # [km] span length
gamma = Ld/Lnl
print('gamma =', gamma)

def f(z):
    return np.exp(-1/La*(Ld*z % Ls))

def F(z):
    k=int(Ld*z/Ls)
    z1 = z - k*Ls/Ld 
    return k*La/Ld*(1-np.exp(-Ls/La)) + La/Ld*(1-np.exp(-Ld/La*z1))

def Leff(z, dz):
    return F(z+dz) - F(z)


def SSFM(u0, dt, dz, L, gamma, order=2, path=True):
    # Initialize solution array and set initial condition
    Nt = len(u0)
    Nz = int(L/dz) + 1
    u = np.zeros((Nz, Nt), dtype=complex)
    u[0, :] = u0
    # Precompute the linear operator
    omega = np.fft.fftfreq(Nt, d=dt) * 2 * np.pi  # Frequency components
    half_linear_operator = np.exp(1j * omega**2/2 * dz / 2)
    linear_operator = np.exp(1j * omega**2/2 * dz)

    if order == 2:
        # Split-step Fourier method: 2th order
        for j in range(1, Nz):
            # Half step of linear part
            u[j, :] = ifft(half_linear_operator * fft(u[j-1, :]))
            # Full step of nonlinear part
            u[j, :] = np.exp(-1j * gamma * Leff((j-1)*dz, dz) * np.abs(u[j, :])**2) * u[j, :]
            # Another half step of linear part
            u[j, :] = ifft(half_linear_operator * fft(u[j, :]))
    elif order == 1:
        # Split-step Fourier method: 1th order
        for j in range(1, Nz):
            # Full step of nonlinear part
            u[j, :] = np.exp(-1j * gamma * Leff((j-1)*dz, dz) * np.abs(u[j-1, :])**2) * u[j-1, :]
            # Full step of linear part
            u[j, :] = ifft(linear_operator * fft(u[j, :])) 

            # u[j, :] = ifft(linear_operator * fft(u[j-1, :]))
            # u[j, :] = np.exp(-1j * gamma * dz * np.abs(u[j, :])**2) * u[j, :]
             
    else:
        raise ValueError('The order must be 1 or 2.')
    
    if path==True:
        return u
    else:
        return u[-1]


def Dispersion(u0, dt, L):
    return SSFM(u0, dt, L, L, gamma=0, order=1, path=False)


def reconstruct(samples, n:int):
    '''
        samples: discrete samples.
        T: sampling period.
        n: points per sample.
    '''
    t = jnp.arange(-10, 10, 1/n)
    sinc = jax.numpy.sinc(t) * jax.numpy.sinc(t/3)   # lanczos kernel
    samples = upsample(samples, n)
    return circFilter(sinc, samples)