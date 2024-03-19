import torch, numpy as np
import scipy.constants as const, scipy.special as special
from src.TorchSimulation.receiver import  BER

def MSE(predict, truth):
    return torch.mean(torch.abs(predict - truth)**2)  # predict, truth: [B, Nmodes]

def SNR(predict, truth):
    return 10 * torch.log10(torch.mean(torch.abs(truth)**2) / torch.mean(torch.abs(predict - truth)**2))

def Qsq(ber):
    return 20 * np.log10(np.sqrt(2) * np.maximum(special.erfcinv(2 * ber), 0.))

def test_model(dataloader, model, loss_fn, device):
    model = model.to(device)
    model.eval()
    mse = 0 
    ber = 0
    power = 0

    N = len(dataloader)
    with torch.no_grad():
        for x, y, z in dataloader:
            x, y, z = x.to(device), y.to(device), z.to(device)
            y_pred = model(x)
            mse += loss_fn(y_pred, y).item()
            power += MSE(0, y).item()
            ber += BER(y, y_pred)['BER']
            
    return {'loss_fn': mse/N, 'BER': np.mean(ber/N), 'SNR': 10 * np.log10(power / mse), 'Qsq': Qsq(np.mean(ber/N)), 'BER_XY':ber/N, 'Qsq_XY': Qsq(ber/N)}