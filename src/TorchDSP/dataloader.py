import pickle, torch, numpy as np, time, random, os, h5py
from .core import TorchInput, TorchSignal, TorchTime
from torch.utils.data import Dataset
import h5py, numpy as np, torch
from src.TorchDSP.loss import Qsq
from src.TorchSimulation.receiver import  BER

def baselines(path, Nch, Rs, P = np.arange(-3, 8)):

    def get_grp(f, Nch, Rs, Pch, Nsymb, NF, SF, L=2000, tag=',method=frequency cut'):
        for key in f.keys():
            if f[key].attrs['Nch'] == Nch and f[key].attrs['Rs(GHz)'] == Rs and f[key].attrs['Pch(dBm)'] == Pch and f[key]['SymbTx'].shape[1] == Nsymb and f[key].attrs['NF(dB)'] == NF and f[key].attrs['freqspace(Hz)']/1e9 / f[key].attrs['Rs(GHz)'] == SF and f[key].attrs['distance(km)'] == L:
                return f[key][f'Rx(sps=2,chid=0{tag})']
            
    def Q_power(f, Nch, pch, Rs,  method='Rx_CDCDDLMS(taps=32,lr=[0.015625, 0.0078125])'):
        grp = get_grp(f, Nch, Rs, pch, 500000, 4.5, 1.2, 2000)
        if 'Qsq' not in grp[method].attrs.keys():
            Q = Qsq(np.mean(BER(torch.from_numpy(grp[method][0])[100000:], torch.from_numpy(grp['Tx'][0,7:-8])[100000:])['BER']))
            grp[method].attrs['Qsq'] = Q
        else:
            Q = grp[method].attrs['Qsq']
        return Q

    
    Q = {}

    with h5py.File(path, 'a') as f:
        Q['CDC'] = [Q_power(f, Nch, pch, Rs, 'Rx_CDCDDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP1'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP1DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP2'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP2DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP4'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP4DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP8'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP8DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP16'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP16DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
        Q['DBP32'] = [Q_power(f, Nch, pch, Rs, 'Rx_DBP32DDLMS(taps=32,lr=[0.015625, 0.0078125])') for pch in P]
    
    return Q



def metric_for(method='CDC',info='Qsq', Nmodes=2, Nch=1, Rs=20, P=0, discard=10000):
    '''
    Get baselines metric from Qfactor dictionary.
    Qfactor saved at _outputs/Qfactor/Nmodes{Nmodes}/{method}.pkl
    method: 'CDC' or 'DBP stps=i' (i=1,2,..,10,20,40)
    discard: 100, 1000, 10000, 20000
    '''
    dic, code = pickle.load(open(f'_outputs/Qfactor/Nmodes{Nmodes}/{method}.pkl','rb'))
    dis = torch.mean(torch.abs(code[:,[0,2,3]] - torch.tensor([P, Rs*2e9,  Nch])), dim=1)
    k = torch.where(dis < 0.1)[0]
    return dic[f'BER from {discard}th symb'][method][info][k]


def get_data(path):
    '''
    Get data from path. Return data, info.
    Example:
        train_data, info = get_data('data/train_data_few.pkl')
        train_data is used to construct signal_dataset.
    '''
    data, info = pickle.load(open(path, 'rb'))
    return data, info


def get_k(Nch, Rs, P, train_t, sps=2) -> list[int]:
    '''
        Return index of batch in train_t infomation with number of channels = Nch, symbol rate = Rs, Power=P.
            train_t: [B, 4].  [P, Fi, Fs, Nch]
    '''
    dis = torch.mean(torch.abs(train_t[:,[0,2,3]] - torch.tensor([P, Rs*sps*1e9,  Nch])), dim=1)
    k = torch.where(dis < 0.1)[0]
    if len(k) == 0:
        print('No matched data')
        raise ValueError
    else:
        # print('match batch: ', k)
        return k.tolist()

def get_k_batch(Nch, Rs, train_t) -> torch.Tensor:
    '''
        Return indexes of batch in train_t infomation with number of channels = Nch, symbol rate = Rs.
            train_t: [B, 4].
    '''
    dis = torch.mean(torch.abs(train_t[:,[2,3]] - torch.tensor([Rs*2e9,  Nch])), dim=1)
    k = torch.where(dis < 0.1)[0]
    if len(k) == 0:
        print('No matched data')
        raise ValueError
    else:
        # print('match batch: ', k)
        return k
    

def get_signals(path: str, Nch: int, Rs: int, Pch=None,  device='cpu', batch_max=10000, idx=(0, None)):
    '''
        Get single mode signals with special Nch and Rs and Pch. 

        Return test_signal, test_truth, test_z.
    '''
    test_y, test_x, test_t = pickle.load(open(path, 'rb'))
    if Pch is not None:
        k = []
        for p in Pch:
            k = k + get_k(Nch, Rs, p, test_t)
    else:
        k = get_k_batch(Nch, Rs, test_t)
    k = k[:batch_max]
    test_signal = TorchSignal(test_y[k, idx[0]:idx[1]], TorchTime(0,0,1)).to(device)
    test_truth = TorchSignal(test_x[k, idx[0]:idx[1]], TorchTime(0,0,1)).to(device)
    test_z = test_t[k].to(device) 
    return test_signal, test_truth, test_z



def get_Qsq(path: str, discard:int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    ''''
    path: path of Q factor dictionary dic.
    discard: number of discard symbols before calculate BER.

    dic:
        - 'BER from nth symb'       n in [0, 1000, 10000, 20000]
            - 'model_path',...
                -  'BER': tensor [B, 1]
                -  'SER': tensor [B, 1]
                -  'Qsq': tensor [B, 1]
                -  'SNR': tensor [B, 1]
        
                B = len(Rs) * len(Nch) * len(P)
    
    return:
        Q, Rs_, Nch_, P_
    '''
    with open(path, 'rb') as f:
        dic, info = pickle.load(f)
    Q = dic[f'BER from {discard}th symb']

    Ps = info[:,0].unique()
    Rss = info[:,2].unique() / 2e9  # [G]
    Nchs = info[:,3].unique()


    res = {}

    for key in Q.keys():
        try:
            Q_ = Q[key]['Qsq'].reshape(len(Rss), len(Nchs), len(Ps), Q[key]['Qsq'].shape[-1])
        except:
            Q_ = Q[key]['Qsq']
        res[key] = Q_
    return res, Rss.numpy(), Nchs.numpy(), Ps.numpy()

def mean_peak(Q: np.ndarray, maxQ = 13):
    '''
        Q: ndarray of Q factor with shape [rs, nch, p]
    '''
    Q = np.minimum(Q, maxQ)  # type: ignore
    return np.mean(np.max(Q, axis=-1))


def getQsq_fromdir(dir: str, discard: int) ->tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    '''
    dir: model path.
    discard: number of discard symbols before calculate BER.
    '''
    Q_path = os.listdir(dir)
    Q_path.sort()
    Qs = {}
    for path in Q_path:
        Qs = {**get_Qsq(dir + path, discard)[0], **Qs}

    _, Rs, Nch, P = get_Qsq(dir + Q_path[0], discard)
    
    return Qs, Rs, Nch, P
    


class signal_dataset:
    """
        task_info: ['P','Fi','Fs','Nch']
        Attributes:
            data: (y, x, w0, a)
                y: np.Array with shape [B, L*sps, Nmodes].  x: np.Array with shape [B, L, Nmodes].  w0: FO value.  a: info dict.
                example of a: {'lpdbm':0, 'carrier_frequency':193e12, 'samplerate':72e9, 'channels': 3}.
            batch_size: int. 
            idx: batch index range.
            i: batch index value.
            sps: samples per symbol of y.
            shffle: bool. True or False.

        Use signal_dataset as a iterator, output a TorchInput each iteration.

        train_data, info = get_data(data_path)   # data_path = 'data/train_data_few.pkl' ...
        dataset = signal_dataset(train_data, batch_size=batch_size, shuffle=True)
        for data in dataset:
            data is TorchInput.
    """
    def __init__(self, data, batch_size=100, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.idx = list(range(self.data.y.shape[0]))
        self.i = 0
        self.sps = self.data.a['sps']
        if shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        return self.data.y.shape[0]
    
    def batch_number(self):
        '''
            return number of batchs.
        '''
        return self.data.y.shape[0] // self.batch_size
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i == self.data.y.shape[0]:
            self.i = 0
            raise StopIteration
        elif self.i + self.batch_size <= self.data.y.shape[0]:
            idx = self.idx[self.i: self.i + self.batch_size]
            self.i += self.batch_size
            signal_input = TorchSignal(val=self.data.y[idx], t=TorchTime(0,0,self.sps))
            signal_output = TorchSignal(val=self.data.x[idx], t=TorchTime(0,0,1))
            task_info = torch.tensor(np.array([self.data.a['lpdbm'][idx], self.data.a['carrier_frequency'][idx], self.data.a['samplerate'][idx], self.data.a['channels'][idx]])).T   # task_info: [batch, 4], info: [P, Fi, Fs, Nch]  Unit:[dBm, Hz, Hz, 1] 
            return TorchInput(signal_input, signal_output, task_info)
        else:
            idx = self.idx[self.i:]
            self.i = self.data.y.shape[0]
            signal_input = TorchSignal(val=self.data.y[idx], t=TorchTime(0,0,self.sps))
            signal_output = TorchSignal(val=self.data.x[idx], t=TorchTime(0,0,1))
            task_info = torch.tensor(np.array([self.data.a['lpdbm'][idx], self.data.a['carrier_frequency'][idx], self.data.a['samplerate'][idx], self.data.a['channels'][idx]])).T
            return TorchInput(signal_input, signal_output, task_info)
        


class opticDataset(Dataset):
    '''
        Framed dataset for optic signal.

        Attributes:
            Nch, Rs, M: number of channels, symbol rate, window size.
            path: data path
            power_fix: if True, the power of signal will be fixed to original launch power.
            y: input signal of shape [B, L*sps, Nmodes].
            x: truth signal with shape [B, L, Nmodes].
            task_info: task infomation with shape [B, 4].  task_info: ['P','Fi','Fs','Nch']


        Use this class as dataset class in torch.
            train_dataset = opticDataset(Nch, Rs, M, path='data/train_data_afterCDCDSP.pkl', power_fix=args.power_fix)
            test_dataset = opticDataset(Nch, Rs, M, path='data/test_data_afterCDCDSP.pkl', power_fix=args.power_fix)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
            for batch_idx, (y, x, info) in enumerate(train_dataloader):
                x = x.to(device)  # [batch_size, Nmodes]
                y = y.to(device)  # [batch_size, M, Nmodes]
    '''
    def __init__(self, Nch, Rs,  M, Pch=None, path='data/train_data_afterCDCDSP.pkl', idx=(0,-1), power_fix=True):
        self.Nch = Nch
        self.Rs = Rs
        self.M = M

        train_y, train_x, train_t = pickle.load(open(path, 'rb'))

        if Pch is not None:
            k = []
            for p in Pch:
                k = k + get_k(Nch, Rs, p, train_t)
        else:
            k = get_k_batch(Nch, Rs, train_t)
            
        if power_fix:
            self.y = train_y[k, idx[0]:idx[1]] * 10**(train_t[k][:,0]/20)[:,None,None]  # [15, 99985, Nmodes]
        else:
            self.y = train_y[k, idx[0]:idx[1]]                                      # [15, 99985, Nmodes]
        self.x = train_x[k, idx[0]:idx[1]]                                          # [15, 99985, Nmodes]
        self.task_info = train_t[k]

    def __len__(self):
        return self.y.shape[0]*(self.y.shape[1] - self.M + 1)
    
    def __getitem__(self, idx):
        i = idx // (self.y.shape[1] - self.M + 1)
        j = idx % (self.y.shape[1] - self.M + 1)
        return self.y[i, j:j+self.M, :], self.x[i, j + (self.M//2), :], self.task_info[i]  # [M, Nmodes], [Nmodes]

'''
h5py dataset:
- pulse
- SymbTx  
- SignalRx
- Rx(sps=2,chid=0,method=frequency cut)
    - info            # [Pch, Fi, Rs, Nch]
    - Tx
    - Rx
    - Rx_CDC 
    - Rx_DBP%d
    - Rx_CDCDDLMS
    - Rx_DBP%dDDLMS
    - ...

'''

class MyDataset(Dataset):

    '''
    window_size = strides + overlaps
    overlaps = |start| + |stop|
    '''

    def __init__(self, path='dataset/test.h5', Nch=[3], Rs=[40], Pch=[-1], Nmodes=2,
                window_size=41, strides=1, Nwindow=10000000, truncate=20000, Tx_window=False,
                rx_grp='Rx(sps=2,chid=0,method=frequency cut)', pre_transform='Rx'):
        # dtype: 'signal' or 'tensor'
        # pre_transform: 'Rx' or 'Rx_CDC' or 'Rx_DBP%d' or 'Rx_CDCDDLMS' or 'Rx_DBP%dDDLMS'
        self.path = path
        self.Nch = Nch if type(Nch) == list else [Nch]
        self.Rs = Rs if type(Rs) == list else [Rs]
        self.Pch = Pch if type(Pch) == list else [Pch]
        self.Nmodes = Nmodes
        self.window_size = window_size
        self.pre_transform = pre_transform
        self.strides = strides
        self.truncate = truncate
        self.Tx_window = Tx_window

        self.Tx = []        # list of  [L, Nmodes]
        self.Rx = []        # list of  [L*sps, Nmodes]
        self.info = []      # list of  [4]

        with h5py.File(path, 'r') as f:
            for key in f.keys():
                group = f[key]
                if group.attrs['Nch'] in self.Nch and group.attrs['Rs(GHz)'] in self.Rs and group.attrs['Nmodes'] == Nmodes and group.attrs['Pch(dBm)'] in self.Pch:

                    if pre_transform not in group[rx_grp].keys():
                        raise ValueError(f"No such pre_transform in {rx_grp}, please choose from {group[rx_grp].keys()}")
                    s = group[rx_grp][pre_transform].attrs['start']
                    e = group[rx_grp][pre_transform].attrs['stop']  + group[rx_grp][pre_transform].shape[1]
                    sps = group[rx_grp][pre_transform].attrs['sps']
                    self.Tx = self.Tx + [torch.from_numpy(group[rx_grp]['Tx'][i,truncate + s:e]).to(torch.complex64) for i in range(group[rx_grp]['Tx'].shape[0])]
                    self.Rx = self.Rx + [torch.from_numpy(group[rx_grp][pre_transform][i, truncate*sps:]).to(torch.complex64)  for i in range(group[rx_grp][pre_transform].shape[0])]
                    self.info = self.info + [torch.from_numpy(group[rx_grp]['info'][i,...]).to(torch.float32)  for i in range(group[rx_grp]['info'].shape[0])]
        
        if self.Tx == []:
            raise ValueError("No such dataset")


        self.Rx_sps = sps
        self.length_list = [(tx.shape[0] - self.window_size)//self.strides + 1 for tx in self.Tx]
        self.length =  min(sum(self.length_list), Nwindow)

    def locate(self,idx):
        '''
        return the index of batch i and the index of window j in the batch.
        '''
        i = np.argmax(np.cumsum(self.length_list) > idx)
        j = int(idx - np.sum(self.length_list[:i]))
        return i, j

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        '''
        Return Rx, Tx, info
        if Tx_window is True,
            shape: [window_size*sps, Nmodes], [windowsize, Nmodes], [4]
        else:
            shape: [window_size*sps, Nmodes], [Nmodes], [4]
            info: [Pch(dBm), Fi(Hz), Fs(Hz), Nch]   power per channel, carrier frequency, samplerate, number of channels
        '''
        i, j = self.locate(idx)

        if self.Tx_window:
            return self.Rx[i][j*self.strides*self.Rx_sps: (j*self.strides+self.window_size)*self.Rx_sps, :], self.Tx[i][j*self.strides:j*self.strides+self.window_size, :], self.info[i]  # [M, Nmodes], [Nmodes]
        else:
            return self.Rx[i][j*self.strides*self.Rx_sps: (j*self.strides+self.window_size)*self.Rx_sps, :], self.Tx[i][j*self.strides+(self.window_size//2), :], self.info[i]
