import pickle , matplotlib.pyplot as plt, torch, numpy as np, argparse, time, os, scipy
from src.TorchDSP.pbc_new import NonlienarFeatures, FoPBC
from src.TorchDSP.dataloader import  get_signals
from src.TorchSimulation.receiver import  BER, nearst_symb



def show_pbc(C, index, index_type='reduce-1', figsize=(5,5),dpi=100, vmax=-1, vmin=-4, s=3):
    x,y = zip(*index)
    values = np.log10(np.abs(C) + 1e-8)
    plt.figure(figsize=figsize, dpi=dpi)
    x = np.array(x)
    y = np.array(y)
    if index_type == 'full':
        plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
    elif index_type == 'reduce-1':
        plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(y, x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
    elif index_type == 'reduce-2':
        plt.scatter(x, y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(y, x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-x, -y, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
        plt.scatter(-y, -x, c=values, cmap='viridis', s=s, vmax=vmax, vmin=vmin)  # `cmap`指定颜色映射，`s`指定点的大小
    plt.colorbar(label='Value')
    plt.xlabel('m Coordinate')
    plt.ylabel('n Coordinate')
    plt.title(f'Heatmap of C_m,n (log10 scale)')


def trainsfrom_signal(path, rho=1, L=400, Nch=1, Rs=40, Pch=[0], index_type='reduce-1', batch_max=10000, idx=(100000, 200000)):
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
    X0 = signal.val[:, idx[0]:idx[1],:]         # [B, N, Nmodes]
    X1 = f(X0, X0, X0)                          # [B, N, Nmodes, p]
    Symb = truth.val[:, idx[0]:idx[1],:]        # [B, N, Nmodes]
    Y = Symb - X0                               # [B, N, Nmodes]
    return P, X0, X1, Y, Symb


def predict(P, X1, C):
    '''
    Input:
        P: [B]
        X1: [B, N, Nmodes, p]
        C: [Nmodes, p]
    Output:
        [B, N, Nmodes]
    '''
    return torch.einsum('bnmp, mp->bnm', X1*P[:,None,None,None] , C)  # [B, N, Nmodes]



def Kernel(P, X1, Y, C, p=2, gamma=1.0, k_type='w=0'):
    '''
    Input:
        P: [B]
        X1:  [B, N, Nmodes, p]
        Y:  [B, N, Nmodes]
        C:  [Nmodes, p]
    Output:
            [B, N, Nmodes]
    '''
    if k_type=='w=0':
        return torch.abs(Y)**(p - 2)
    elif k_type=='w!=0':
        return torch.abs(Y - predict(P, X1, C))**(p - 2)
    elif k_type=='p-gamma':
        return torch.minimum(torch.ones(()), gamma / torch.abs(Y))**2 * torch.abs(Y)**(p - 2)
    else:
        raise ValueError('k_type should be w=0 or w!=0')



def fit(P, X1, Y, weight=None, lamb_l2:float=0, pol_sep=True):
    '''
    fit the coeff for PBC, if pol_sep==True, then the PBC is applied to each polarization separately
    Input:
        P: [B]
        X: [B, N, Nmodes, p]    Rx
        Y: [B, N, Nmodes]
        weight: [B, N, Nmodes]
    output:
        C: [Nmodes, p]

         C = argmin_{C}  \sum weight*|X1 @ C - Y|^2
    '''
    X1= X1 * P[:,None,None,None]  

    if weight == None:
        weight = torch.ones_like(Y)

    Nmodes = Y.shape[-1]

    if pol_sep:
        A = torch.einsum('bnm, bnmp, bnmq -> mpq', weight.to(torch.complex64), X1.conj(), X1) / X1.shape[1]  # [m, p, p]
        b = torch.einsum('bnm, bnmp, bnm  -> mp', weight.to(torch.complex64), X1.conj(), Y) / X1.shape[1]    # [m, p]
        C = [torch.linalg.solve(A[i] + lamb_l2 * torch.eye(A.shape[-1]), b[i]) for i in range(A.shape[0])]
        return torch.stack(C, dim=0)
    else:
        A = torch.einsum('bnm, bnmp, bnmq -> pq', weight.to(torch.complex64), X1.conj(), X1) / X1.shape[1]   # [p, p]
        b = torch.einsum('bnm, bnmp, bnm  -> p', weight.to(torch.complex64), X1.conj(), Y) / X1.shape[1]    # [p]
        C = torch.linalg.solve(A + lamb_l2 * torch.eye(A.shape[-1]), b)
        return torch.stack([C]*Nmodes, dim=0)



def test(data, C, BER_discard=20000):
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
    pbc = predict(P, X1, C) # [B, N, Nmodes]
    Yhat = X0 +  pbc   # [B, N, Nmodes] 
    metric = BER(Yhat[:,BER_discard:,:], Symb[:,BER_discard:,:])
    return metric


if __name__ == '__main__':
    # train_path = "data/train_data_afterCDCDSP.pkl"
    # train_path = "data/Nmodes1/train_data_afterCDCDSP.pkl"
    # test_path = "data/test_data_afterCDCDSP.pkl"
    # test_path = "data/Nmodes1/test_afterCDCDSP.pkl"
    path = "data/lab/test_2_0_4_1.pkl"
    Nmodes = 2
    Nch = 7
    Rs = 36
    train_path = path 
    test_path = path
    idx1 = (100000, 200000)
    idx2 = (200000, 300000)


    # train_path = "data/Nmodes2/train_afterCDCDSP.pkl"
    # test_path = "data/Nmodes2/test_afterCDCDSP.pkl"
    # Nmodes = 2
    # Nch = 5 
    # Rs = 40
    # idx1 = (10000, -10000)
    # idx2 = (10000, -10000)

    rho = 1
    L = 200
    index_type = 'reduce-1'
    train_p = [0]
    test_p = [0]
    lamb_l2 = 0

    index = NonlienarFeatures(Nmodes=Nmodes, rho=rho, L = L, index_type=index_type).index
    train_data = trainsfrom_signal(train_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=train_p, index_type=index_type, idx=idx1)
    test_data = trainsfrom_signal(test_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=test_p, index_type=index_type, idx=idx2)

    s, e = 1000, -1000
    use_Rx_only = True
    P, X0, X1, Y_real, Symb = train_data
    Y = nearst_symb(X0) - X0 if use_Rx_only else Y_real
    C0 =  fit(P, X1[:,s:e,:,:], Y[:,s:e,:], lamb_l2=lamb_l2, pol_sep=True)   # PBC 系数

    ps = np.linspace(1, 3, 11)
    gammas = np.linspace(0.1, 4, 40)
    Q_list = {}
    L_list = {}


    for p in ps:
        for gamma in gammas:
            t0 = time.time()
            weight = Kernel(P, X1[:,s:e,:,:], Y[:,s:e,:], C0, p=p, gamma=torch.std(Y).item()*gamma, k_type='p-gamma')   # RKN kenel


            C = fit(P, X1[:,s:e,:,:], Y[:,s:e,:], weight, lamb_l2=lamb_l2, pol_sep=False)  # [Nmodes, p]   X1: 非线性特征    Y: 误差
            #  C = argmin_{C}  \sum weight*|X1 @ C - Y|^2


            L1, Q1 = test(test_data, C, BER_discard=200)
            t1 = time.time()
            print(f'p={p}, gamma={gamma}, time: {t1-t0}, Q1={Q1}, MSE={L1}')
            Q_list[f'p={p}, gamma={gamma}'] = Q1
            L_list[f'p={p}, gamma={gamma}'] = L1


    pickle.dump((Q_list, L_list), open(f'res_p_gamma.pkl', 'wb'))


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
# res = {}
# Ls = [100, 200, 300, 400, 500, 600, 700, 800]
# Ns = [50, 100, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000]



# train_path = "data/Nmodes1/train_data_afterCDCDSP.pkl"
# test_path = "data/test_data_afterCDCDSP.pkl"
# Nch = 1
# Rs = 40
# rho = 1
# train_ps = [[0], [-2], [-1]]
# test_ps = [[-2], [-1], [0]]
# lamb_l2 = 0

# for train_p in train_ps:
#     for L in Ls:
#         train_data = trainsfrom_signal(train_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=train_p, index_type='reduce-1', batch_max=7)
#         for n in Ns:
#             s, e = 50000-n, 50000+n
#             P, X0, X1, Y, Symb = train_data
#             C = fit(P, X1[:,s:e,:,:], Y[:,s:e,:], lamb_l2=lamb_l2, pol_sep=False)  # [Nmodes, p]

#             for test_p in test_ps:
#                 test_data = trainsfrom_signal(test_path, rho=rho, L=L, Nch=Nch, Rs=Rs, Pch=test_p, index_type='reduce-1')
#                 Q1, L1, Qs, ls, xis = test_PBC(C, test_data)
#                 print(f'train_p={train_p}, test_p={test_p}, L={L}, D={2*n*7}')
#                 res[f'Q1 D={2*n*7} L={L} train_p={train_p} test_p={test_p}'] = Q1
#                 res[f'xis D={2*n*7} L={L} train_p={train_p} test_p={test_p}'] = xis
#                 res[f'MSE D={2*n*7} L={L} train_p={train_p} test_p={test_p}'] = L1
#                 res[f'Ls D={2*n*7} L={L} train_p={train_p} test_p={test_p}'] = ls
#                 res[f'Qs D={2*n*7} L={L} train_p={train_p} test_p={test_p}'] = Qs


# print(res)
# pickle.dump(res, open(f'res_D_L_large.pkl', 'wb'))


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
