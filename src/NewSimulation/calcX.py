from .config import *
import pickle
import torch, time
from tqdm import tqdm


def calc_Xcoeff(dz, z_max, dt, t_max, Ts, N, device='cuda:0'):

    zsteps = int(z_max/dz)
    # initial condition
    t = np.arange(-t_max, t_max, dt)
    u0 = np.sinc(t/Ts)

    us = []
    us.append(u0)
    for i in range(zsteps):
        u = Dispersion(us[-1], dt, dz)
        us.append(u)

    gc = np.stack(us, axis=0)
    gc = torch.tensor(gc).to(device)

    steps = int(Ts/dt)
    X = torch.zeros(zsteps+1, 2*N+1, 2*N+1, 2*N+1, dtype=torch.complex64).to(device)
    for m in tqdm(range(-N, N+1)):
        for n in range(-N, N+1):
            for k in range(-N, N+1):
                X[:,m, n, k] = torch.sum(torch.roll(gc, m*steps, dims=-1)*torch.roll(gc, n*steps, dims=-1)*torch.roll(gc, k*steps, dims=-1).conj()*gc.conj()*dt, dim=-1)
    return X, torch.arange(0, z_max+dz, dz)


t0 = time.time()
X,z = calc_Xcoeff(1, 16, 0.1, 100, 0.5, 80)
t1 = time.time()
print('Time cost for Xcoeff_161_dz1_zmax16:', t1-t0)
pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_161_dz1_zmax16.pkl', 'wb'))

t0 = time.time()
X,z = calc_Xcoeff(0.25, 16, 0.1, 100, 0.5, 80)
t1 = time.time()
print('Time cost for Xcoeff_161_dz0.25_zmax16:', t1-t0)
pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_161_dz0.25_zmax16.pkl', 'wb'))

