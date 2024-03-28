'''
    Loss functions for training.
'''
import torch, numpy as np
import scipy.constants as const, scipy.special as special

# define loss function
def well(x, mu=4):
    return torch.sigmoid(mu*(x - 0.3162)) + torch.sigmoid(mu*(-x - 0.3162))

def BER_well(predict, truth, mu=1):
    error = predict - truth
    dis = torch.max(torch.abs(error.real), torch.abs(error.imag))
    return torch.mean(well(dis, mu))


def MSE(predict, truth, mu=0):
    return torch.mean(torch.max(torch.abs(predict - truth)**2, torch.tensor(mu)))  # predict, truth: [B, Nmodes]


def SNR(predict, truth):
    '''
    Signal noise ratio.
    '''
    return 10 * torch.log10(torch.mean(torch.abs(truth)**2) / torch.mean(torch.abs(predict - truth)**2))

def Qsq(ber):
    '''
    Q factor.
    '''
    return 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * ber), 0.))