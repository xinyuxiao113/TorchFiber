import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os, scipy
from src.TorchDSP.pbc_new import NonlienarFeatures, FoPBC
from src.TorchDSP.dataloader import  get_signals
from src.TorchSimulation.receiver import  BER

def fit(X,Y, lamb_l2:float=0):
    A = (X.T.conj() @ X) / X.shape[0] + lamb_l2 * torch.eye(X.shape[1])               # A.eig: 4e5 ~ 1e8
    b = (X.T.conj()) @ Y /X.shape[0]
    return torch.linalg.solve(A, b)

def test(xis, X_train, symb_train, C, power_diff, k=1, BER_discard=10000):
    pbc = X_train[:,k:] @ C
    Ls = []
    Qs = []

    for xi in xis:
        Yhat0 = X_train[:,0] + xi*10**(power_diff/10)*pbc
        Ls.append(torch.mean(torch.abs(Yhat0 - symb_train)**2))
        Qs.append(BER(Yhat0[BER_discard//2:-BER_discard//2,None], symb_train[BER_discard//2:-BER_discard//2, None])['Qsq'])
    return np.array(Ls), np.array(Qs)

train_path = "data/train_data_afterCDCDSP.pkl"
test_path = "data/test_data_afterCDCDSP.pkl"


train_p = -2
test_p = -2
train_signal, train_truth, train_z = get_signals(train_path, Nch=1, Rs=40, Pch=[train_p],  device='cpu')
test_signal, test_truth, test_z = get_signals(test_path, Nch=1, Rs=40, Pch=[test_p], device='cpu')

def test_PBC(rho, L, lamb_l2=0.1, s=2000, e=-2000, k=1):
    '''
        s,e,k = 2000, -2000, 1            # s,e:symol start and end, k:feature start 0 or 1
        lamb_l2 = 0
    '''
    f = NonlienarFeatures(Nmodes=1, rho=rho, L = L, index_type='reduce-1')
    X_train = torch.cat([torch.squeeze(train_signal.val)[:, None], torch.squeeze(f(train_signal.val, train_signal.val, train_signal.val))], dim=1)
    X_test  = torch.cat([torch.squeeze(test_signal.val)[:, None], torch.squeeze(f(test_signal.val, test_signal.val, test_signal.val))], dim=1)
    Y_train = torch.squeeze(train_truth.val - train_signal.val)
    Y_test = torch.squeeze(test_truth.val - test_signal.val)
    symb_train = torch.squeeze(train_truth.val)
    symb_test = torch.squeeze(test_truth.val)

    C = fit(X_train[s:e,k:], Y_train[s:e], lamb_l2=lamb_l2)

    xis = np.linspace(0.6, 1.5, 100)
    Ls, Qs = test(xis, X_test, symb_test, C, power_diff=test_p-train_p, k=k)
    Q1 = test([1], X_test, symb_test, C, power_diff=test_p-train_p, k=k)[1][0,0]

    return np.max(Qs), Q1, xis[np.argmax(Qs)]
# Ls_ = (Ls - np.mean(Ls)) / np.std(Ls)
# Qs_ = (Qs - np.mean(Qs)) / np.std(Qs)

# plt.plot(xis, -Ls_)
# plt.plot(xis, Qs_)
# print('best xi for Q factor', xis[np.argmax(Qs)], '   Best Q factor:', np.max(Qs),  '       xi=1 Q factor:', Q1 )
# print('best xi for Q factor', xis[np.argmin(Ls)])
# print(f'|{rho}|{L}| {train_p} | {test_p} | [{s}:{e}] | {lamb_l2} | {Q1:.2f} (xi=1)| {np.max(Qs):.2f} (xi={xis[np.argmax(Qs)]:.2f})|')


# test rho, L
# res = {}
# for rho in [0.5, 1, 1.5, 2]:
#     for L in [100, 200, 300, 400, 500, 600, 700, 800]:
#         t0 = time.time()
#         Qm, Q1,xi = test_PBC(rho, L, lamb_l2=0.01, s=2000, e=-2000, k=1)
#         t1 = time.time()
#         print(f'rho={rho}, L={L}, time: {t1-t0}')
#         res[f'Qmax rho={rho}, L={L}'] = Qm
#         res[f'Q1 rho={rho}, L={L}'] = Q1
#         res[f'xi rho={rho}, L={L}'] = xi

# print(res)
# pickle.dump(res, open('res.pkl', 'wb'))


# test lamb_l2
# lamb_l2s = 10**(np.linspace(-3, -1, 10))
# res = {} 
# for lamb_l2 in lamb_l2s:
#     t0 = time.time()
#     Qm, Q1,xi = test_PBC(1, 800, lamb_l2=lamb_l2, s=2000, e=-2000, k=1)
#     t1 = time.time()
#     print(f'lamb_l2={lamb_l2}, time: {t1-t0}')
#     res[f'Qmax lamb_l2={lamb_l2}'] = Qm
#     res[f'Q1 lamb_l2={lamb_l2}'] = Q1
#     res[f'xi lamb_l2={lamb_l2}'] = xi

# print(res)
# pickle.dump(res, open('res_lamb_l2.pkl', 'wb'))

# test train data size
# res = {}
# Ls = [100, 200, 300, 400, 500, 600, 700, 800]
# Ns = [500, 1000, 2000, 4000, 8000, 16000, 32000, 48000]

# for L in Ls:
#     for n in Ns:
#         Qm, Q1,xi = test_PBC(1, L, lamb_l2=0, s=50000-n, e=50000+n, k=1)
#         res[f'Qmax D={2*n} L={L}'] = Qm
#         res[f'Q1 D={2*n} L={L}'] = Q1
#         res[f'xi D={2*n} L={L}'] = xi
#         print(Qm, Q1, xi)

# print(res)
# pickle.dump(res, open(f'res_D_L_Tr{train_p}_Te{test_p}.pkl', 'wb'))


n = 48000
Qm, Q1,xi = test_PBC(1, 800, lamb_l2=0, s=50000-n, e=50000+n, k=1)
print(Qm, Q1, xi)