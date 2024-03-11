import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os, scipy
from src.TorchDSP.pbc_new import NonlienarFeatures, FoPBC
from src.TorchDSP.dataloader import  get_signals
from src.TorchSimulation.receiver import  BER


def trainsfrom_signal(path, rho=1, L=400, Nch=1, Rs=40, Pch=[0], index_type='reduce-1', batch_max=10000):
    '''
    Input:
        signal.val [B, N, Nomdes]
        truth.val [B, N, Nomdes]
        z [B, 1]
    Ouput:
        P [B]
        X0 [B, N, Nmodes]
        X1 [B, N, Nomdes, p]
        Y [B, N, Nomdes]
        Symb [B, N, Nomdes]
    '''
    signal, truth, z = get_signals(path, Nch=Nch, Rs=Rs, Pch=Pch,  device='cpu', batch_max=batch_max)
    f = NonlienarFeatures(Nmodes=signal.val.shape[-1], rho=rho, L = L, index_type=index_type)
    P = 10**(z[:,0]/10)                         # [B] 
    X0 = signal.val                             # [B, N, Nmodes]
    X1 = f(signal.val, signal.val, signal.val)  # [B, N, Nmodes, p]
    Symb = truth.val                            # [B, N, Nmodes]
    Y = truth.val - signal.val                  # [B, N, Nmodes]
    return P, X0, X1, Y, Symb



def fit(P, X, Y, lamb_l2:float=0, pol_sep=False):
    '''
    fit the coeff for PBC, if pol_sep==True, then the PBC is applied to each polarization separately
    Input:
        P: [B]
        X: [B, N, Nmodes, p]  
        Y: [B, N, Nmodes]
    output:
        C: [Nmodes, p]
    '''
    X = X * P[:,None,None,None]  
    weight = torch.ones(P.shape[0]) / P.shape[0]
    Nmodes = Y.shape[-1]

    if pol_sep:
        A = torch.einsum('b, bnmp,bnmq -> mpq', weight.to(torch.complex64), X.conj(), X) / X.shape[1]  # [m, p, p]
        b = torch.einsum('b, bnmp, bnm -> mp', weight.to(torch.complex64), X.conj(), Y) / X.shape[1]  # [m, p]
        C = [torch.linalg.solve(A[i] + lamb_l2 * torch.eye(A.shape[-1]), b[i]) for i in range(A.shape[0])]
        return torch.stack(C, dim=0)
    else:
        A = torch.einsum('b, bnmp, bnmq -> pq', weight.to(torch.complex64), X.conj(), X) / X.shape[1]  # [p, p]
        b = torch.einsum('b, bnmp, bnm -> p', weight.to(torch.complex64), X.conj(), Y) / X.shape[1]   # [p]
        C = torch.linalg.solve(A + lamb_l2 * torch.eye(A.shape[-1]), b)
        return torch.stack([C]*Nmodes, dim=0)



def test(data, C, xis, BER_discard=10000):
    '''
        xis: [L]
        P, X0, X1, Y, Symb = *data 
            P [B]
            X0 [B, N, Nmodes]
            X1 [B, N, Nomdes, p]
            Y [B, N, Nomdes]
            Symb [B, N, Nomdes]
        C:  [Nmodes, p]
    '''
    Ls, Qs = [], []

    P, X0, X1, Y, Symb = data
    pbc = torch.einsum('bnmp, mp->bnm', X1*P[:,None,None,None] , C) # [B, N, Nmodes]


    for xi in xis:
        Yhat = X0 + xi * pbc   # [B, N, Nmodes] 
        metric = BER(Yhat[:,BER_discard//2:-BER_discard//2,:], Symb[:,BER_discard//2:-BER_discard//2,:])
        Ls.append(torch.mean(torch.abs(Yhat -  Symb)**2, dim=1))  # [B, Nmodes]
        Qs.append(metric['Qsq'])                                  # [B, Nmodes]
    return np.array(Ls), np.array(Qs)   # [L, B, Nmodes], [L, B, Nmodes]


def test_PBC(train_data, test_data, lamb_l2=0.1, s=2000, e=-2000, xis=np.linspace(0.6, 1.5, 100)):
    '''
        s,e,k = 2000, -2000, 1            # s,e:symol start and end, k:feature start 0 or 1
        lamb_l2 = 0
    '''
    P, X0, X1, Y, Symb = train_data
    C = fit(P, X1[:,s:e,:,:], Y[:,s:e,:], lamb_l2=lamb_l2, pol_sep=False)  # [Nmodes, p]

    Ls, Qs = test(test_data, C, xis=xis)
    L1, Q1 = test(test_data, C, xis=np.ones(1))
    return Q1, L1, Qs, Ls, xis  # [1, B, Nmodes], [1, B, Nmodes], [L, B, Nmodes], [L, B, Nmodes], [L]


# # train_path = "data/train_data_afterCDCDSP.pkl"
# test_path = "data/test_data_afterCDCDSP.pkl"
# train_path = "data/Nmodes1/train_afterCDCDSP.pkl"
# # test_path = "data/Nmodes1/test_afterCDCDSP.pkl"

# # Fix: Nch=1, Rs=40
# Nch = 1
# Rs = 40

# # parameters: rho, L, train_p, test_p, D = 2n, lamb_l2
# rho = 1
# L = 400 
# train_p = [-1]
# test_p = [-2]
# n = 48000
# lamb_l2 = 0


# t0 = time.time()
# train_data = trainsfrom_signal(train_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=train_p, index_type='reduce-1', batch_max=7)
# test_data = trainsfrom_signal(test_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=test_p, index_type='reduce-1')
# Q1, L1, Qs, Ls, xis = test_PBC(train_data, test_data, lamb_l2=lamb_l2, s=50000-n, e=50000+n)
# t1 = time.time()

# print(f'train_p={train_p}, test_p={test_p}, rho={rho}, L={L}')
# print(f'number of signals = {train_data[0].shape[0]}')
# print(f'time: {t1-t0}')
# print(Q1.shape, L1.shape, Qs.shape, Ls.shape, xis.shape)
# print(Q1)


# Ls_ = (Ls - np.mean(Ls)) / np.std(Ls)
# Qs_ = (Qs - np.mean(Qs)) / np.std(Qs)

# plt.plot(xis, -Ls_)
# plt.plot(xis, Qs_)
# print('best xi for Q factor', xis[np.argmax(Qs)], '   Best Q factor:', np.max(Qs),  '       xi=1 Q factor:', Q1 )
# print('best xi for Q factor', xis[np.argmin(Ls)])
# print(f'|{rho}|{L}| {train_p} | {test_p} | [{s}:{e}] | {lamb_l2} | {Q1:.2f} (xi=1)| {np.max(Qs):.2f} (xi={xis[np.argmax(Qs)]:.2f})|')


# train_p = 0
# test_p = -2
# train_data = get_signals(train_path, Nch=1, Rs=40, Pch=[train_p],  device='cpu')
# test_data = get_signals(test_path, Nch=1, Rs=40, Pch=[test_p], device='cpu')


# # test 1: rho, L
# res = {}
# for rho in [0.5, 1, 1.5, 2]:
#     for L in [100, 200, 300, 400, 500, 600, 700, 800]:
#         t0 = time.time()
#         Qm, Q1,xi, L1, Ls, Qs = test_PBC(rho, L, train_data, test_data, lamb_l2=0.01, s=2000, e=-2000, k=1)
#         t1 = time.time()
#         print(f'rho={rho}, L={L}, time: {t1-t0}')
#         res[f'Qmax rho={rho}, L={L}'] = Qm
#         res[f'Q1 rho={rho}, L={L}'] = Q1
#         res[f'xi rho={rho}, L={L}'] = xi
#         res[f'MSE rho={rho}, L={L}'] = L1
#         res[f'Ls rho={rho}, L={L}'] = Ls 
#         res[f'Qs rho={rho}, L={L}'] = Qs

# print(res)
# pickle.dump(res, open(f'res_rho_L_Tr{train_p}_Te{test_p}.pkl', 'wb'))


# # test 2: lamb_l2
# lamb_l2s = 10**(np.linspace(-3, -1, 10))
# res = {} 
# for lamb_l2 in lamb_l2s:
#     t0 = time.time()
#     Qm, Q1,xi, L1, Ls, Qs = test_PBC(1, 800,train_data, test_data, lamb_l2=lamb_l2, s=2000, e=-2000, k=1)
#     t1 = time.time()
#     print(f'lamb_l2={lamb_l2}, time: {t1-t0}')
#     res[f'Qmax lamb_l2={lamb_l2}'] = Qm
#     res[f'Q1 lamb_l2={lamb_l2}'] = Q1
#     res[f'xi lamb_l2={lamb_l2}'] = xi
#     res[f'MSE lamb_l2={lamb_l2}'] = L1
#     res[f'Ls lamb_l2={lamb_l2}'] = Ls
#     res[f'Qs lamb_l2={lamb_l2}'] = Qs

# print(res)
# pickle.dump(res, open(f'res_lamb_l2_Tr{train_p}_Te{test_p}.pkl', 'wb'))

# # test 3: train data size
res = {}
Ls = [100, 200, 300, 400, 500, 600, 700, 800]
Ns = [50, 100, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000]

train_path = "data/Nmodes1/train_afterCDCDSP.pkl"
test_path = "data/test_data_afterCDCDSP.pkl"
Nch = 1
Rs = 40
rho = 1
train_p = [0]
test_p = [-2]
lamb_l2 = 0


for L in Ls:
    for n in Ns:
        t0 = time.time()
        train_data = trainsfrom_signal(train_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=train_p, index_type='reduce-1', batch_max=7)
        test_data = trainsfrom_signal(test_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=test_p, index_type='reduce-1')
        Q1, L1, Qs, Ls, xis = test_PBC(train_data, test_data, lamb_l2=lamb_l2, s=50000-n, e=50000+n)
        t1 = time.time()
        print(f'D={2*n*7}, L={L}, time: {t1-t0}, Q1: {Q1}')
        res[f'Q1 D={2*n*7} L={L}'] = Q1
        res[f'xis D={2*n*7} L={L}'] = xis
        res[f'MSE D={2*n*7} L={L}'] = L1
        res[f'Ls D={2*n*7} L={L}'] = Ls
        res[f'Qs D={2*n*7} L={L}'] = Qs

print(res)
pickle.dump(res, open(f'res_D_L_Tr{train_p}_Te{test_p}_large.pkl', 'wb'))


# test 4: train power and test power
# train_p, test_p, rho, L, lamb_l2, n, k, Rs, Nch
# train_ps = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,4,5,6]
# test_ps =  [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,4,5,6]

# rho = 1
# L = 400
# n = 48000
# test_data = trainsfrom_signal(test_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=test_ps, index_type='reduce-1')
# res = {}

# for train_p in train_ps:
#     t0 = time.time()
#     train_data = trainsfrom_signal(train_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=[train_p], index_type='reduce-1')
#     Q1, L1, Qs, Ls, xis = test_PBC(train_data, test_data, lamb_l2=0, s=50000-n, e=50000+n)
#     t1 = time.time()
#     print(f'train_p={train_p}, time: {t1-t0}')
#     res[f'Q1 tr={train_p}'] = Q1
#     res[f'MSE tr={train_p}'] = L1
#     res[f'xis tr={train_p}'] = xis
#     res[f'Ls tr={train_p}'] = Ls
#     res[f'Qs tr={train_p}'] = Qs

# print(res)
# pickle.dump(res, open(f'res_TrTe_L{L}_D{2*n}.pkl', 'wb'))
