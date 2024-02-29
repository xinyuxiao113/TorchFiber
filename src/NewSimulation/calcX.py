from .config import *
import pickle
import torch, time
from tqdm import tqdm


def calc_Xcoeff(dz, z_max, dt, t_max, Ts, N, device='cuda:0', phase_matching=False):
    '''
    dz: step size of z.  unit [Ld]
    z_max: max value of z.  unit [Ld]
    dt: step size of t.      unit [T0]
    t_max: max value of t.  unit [T0]
    Ts: period of the signal. unit [T0]
    N: number of coefficients. 
    device: device to store the tensor
    '''

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

    if phase_matching == False:
        X = torch.zeros(zsteps+1, 2*N+1, 2*N+1, 2*N+1, dtype=torch.complex64).to(device)
        for m in tqdm(range(-N, N+1)):
            for n in range(-N, N+1):
                for k in range(-N, N+1):
                    X[:,m, n, k] = torch.sum(torch.roll(gc, m*steps, dims=-1)*torch.roll(gc, n*steps, dims=-1)*torch.roll(gc, k*steps, dims=-1).conj()*gc.conj()*dt, dim=-1)
        return X, torch.linspace(0, z_max, zsteps+1)
    else:
        X = torch.zeros(zsteps+1, 2*N+1, 2*N+1, dtype=torch.complex64).to(device)
        for m in tqdm(range(-N, N+1)):
            for n in range(-N, N+1):
                X[:,m, n] = torch.sum(torch.roll(gc, m*steps, dims=-1)*torch.roll(gc, n*steps, dims=-1)*torch.roll(gc, (m+n)*steps, dims=-1).conj()*gc.conj()*dt, dim=-1)
        return X, torch.linspace(0, z_max, zsteps+1)
        


t0 = time.time()
X,z = calc_Xcoeff(0.01, 2, 0.01, 100, 0.5, 80, phase_matching=True)
t1 = time.time()
print('Time cost for Xcoeff_161_dz0.01_zmax2_Ts0.5_2D:', t1-t0)
pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_161_dz0.01_zmax2_Ts0.5_2D.pkl', 'wb'))


# t0 = time.time()
# X,z = calc_Xcoeff(0.1, 2, 0.1, 100, 0.5, 10)
# t1 = time.time()
# print('Time cost for Xcoeff_21_dz0.1_zmax2_Ts0.5:', t1-t0)
# pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_21_dz0.1_zmax2_Ts0.5.pkl', 'wb'))


# t0 = time.time()
# X,z = calc_Xcoeff(0.05, 2, 0.1, 100, 0.5, 10)
# t1 = time.time()
# print('Time cost for Xcoeff_21_dz0.05_zmax2_Ts0.5:', t1-t0)
# pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_21_dz0.05_zmax2_Ts0.5.pkl', 'wb'))


# t0 = time.time()
# X,z = calc_Xcoeff(1, 16, 0.1, 100, 0.5, 80)
# t1 = time.time()
# print('Time cost for Xcoeff_161_dz1_zmax16:', t1-t0)
# pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_161_dz1_zmax16.pkl', 'wb'))

# t0 = time.time()
# X,z = calc_Xcoeff(0.25, 16, 0.1, 100, 0.5, 80)
# t1 = time.time()
# print('Time cost for Xcoeff_161_dz0.25_zmax16:', t1-t0)
# pickle.dump((X.to('cpu').numpy(), z.to('cpu').numpy()), open('data/Xcoeff_161_dz0.25_zmax16.pkl', 'wb'))

