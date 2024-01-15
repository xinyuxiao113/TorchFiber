# type: ignore
import jax.random as rd, optax, numpy as np , os
from .core import MySignal, SigTime
from .utils import HiddenPrints
from .receiver import get_dataset, merge_dataset, split_dataset
from .models import CDC_transform, DBP_transform

# 使用os检查是否存在path
path_ = '/home/xiaoxinyu/data'                                   # data path on server
if not os.path.exists(path_): path_ = '/Users/xiaoxinyu/Desktop/data'     # data path on server
if not os.path.exists(path_): raise(ValueError)                  # 请设置正确的数据路径


class DataLoader():
    def __init__(self,ds, ds_info, batchsize, overlaps, count=0):
        self.ds_info = ds_info
        self.ds = ds
        self.batchsize = batchsize
        self.overlaps = overlaps
        self.sps = int(ds.a['sps'])
        
        self.flen = self.batchsize + self.overlaps
        self.Nx = ds.x.shape[-2]
        self.Ny = ds.y.shape[-2]


        self.idx = np.arange(self.flen)
        self.idy = np.arange(self.flen * self.sps)

        self.count = count

    def get_data(self):
        if self.ds == None: self.load_data()
        i = self.count
        x = MySignal(self.ds.x[...,(self.idx + i* self.batchsize) % self.Nx,:],      SigTime(0,0,1),   self.ds.a['samplerate']/self.sps, self.ds.a['lpdbm'], self.ds.a['carrier_frequency'], self.ds.a['channels'])  # type: ignore
        y = MySignal(self.ds.y[...,(self.idy + i* self.batchsize*self.sps) % self.Ny,:], SigTime(0,0,self.sps), self.ds.a['samplerate'], self.ds.a['lpdbm'], self.ds.a['carrier_frequency'], self.ds.a['channels'])
        self.count += 1
        return y,x
    
    def remove_data(self):
        self.ds = None
    
    def load_data(self):   
        self.ds, _ = Generate_Data(**self.ds_info)



def Generate_Data(mark='A_batch2', Nch=[1], Rs=[80], SF=1.1, mode=1, batch_id=[0], power=[-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6], path=None, merge=False, transform=None, dbpsteps=None):
    '''
        transform: None, CDC, DBP
        Nch: list or int.
        Rs: list or int.
        power: list of list or list of int.
    '''
    if path==None:
        path = path_ + f'/{mark}_bits4e5_SF{SF}_mode{mode}'

    # 获取函数的所有局部变量
    variables = locals()
    # 过滤掉一些特殊变量，例如 '__builtins__'、'self'、'cls' 等
    variables = {k: v for k, v in variables.items() if not k.startswith('__') and not callable(v)}

    ## Gloabal setting
    key = rd.PRNGKey(234)
    rx_sps = 2
    FO = 0
    lw = 0 

    data_list = []

    if type(Rs) != list: Rs = [Rs]
    if type(Nch) != list: Nch = [Nch]

    
    with HiddenPrints():
        for i,rs in enumerate(Rs):
            for j,nch in enumerate(Nch):
                if type(power[0]) !=list:
                    for p in power: 
                        chid = nch // 2
                        path_tx = path + f'/Tx_Nch{nch}_{rs}GHz_Pch{p}dBm'
                        path_rx = path + f'/Channel_Nch{nch}_{rs}GHz_Pch{p}dBm'
                        data = get_dataset(path_tx,path_rx,key, batch_id, chid, rx_sps, FO, lw)
                        data_list.append(data)
                else:
                    for p in power[i]:
                        chid = nch // 2
                        path_tx = path + f'/Tx_Nch{nch}_{rs}GHz_Pch{p}dBm'
                        path_rx = path + f'/Channel_Nch{nch}_{rs}GHz_Pch{p}dBm'
                        data = get_dataset(path_tx,path_rx,key, batch_id, chid, rx_sps, FO, lw)
                        data_list.append(data)

    data_list = merge_dataset(data_list)
    if transform == 'CDC': data_list = CDC_transform(data_list )
    if transform == 'DBP': data_list = DBP_transform(data_list, dbpsteps)

    # merge or not
    if not merge:
        data_list = split_dataset(data_list)

    return data_list, variables

