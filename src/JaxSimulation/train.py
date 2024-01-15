
# type: ignore
import numpy as np,  matplotlib.pyplot as plt, jax, jax.numpy as jnp, pandas, optax, flax.linen as nn
from functools import partial
from typing import Callable
import cloudpickle
from tqdm import tqdm

import src.JaxSimulation.DataLoader as DL
from .receiver import BER, SER, DataInput
from .core import wrap_signal
from .dsp import init_model, TrainState, construct_update





class Model:
    
    model_info : dict
    train_device: str='cpu'      #  'cpu' or 'gpu'.
    test_device: str='cpu'
    net_train: nn.Module
    net_test: nn.Module
    apply: Callable
    ol:int
    optimizer: optax.GradientTransformation
    TS:TrainState
    update_step: Callable
    dataloader: DL.DataLoader


    @staticmethod
    def load(path: str):
        model = cloudpickle.load(open(path, 'rb'))
        if model.dataloader.ds == None: model.dataloader.load_data()

        model.net_train = init_model(model.model_info, mode='train', initialization=False, batch_dim=True)
        model.net_test = init_model(model.model_info, mode='test', initialization=False, batch_dim=False)
        model.apply = jax.jit(partial(model.net_test.apply, mutable='state'), static_argnums=3, backend=model.test_device)
        model.update_step = construct_update(model.net_train, model.optimizer, device=model.train_device)
        print('load model complete. remember to update data before training.')
        return model
        
        
    def init(self, model_info, batch_dim=2, train_device='cpu', test_device='cpu', update_state=False, data_train: DataInput=None):
        '''
        Intialize model, optimizer, and update function.
            Input:
                model_info: dict
                train_device: str. 'cpu' or 'gpu'.
                test_device: str. 'cpu' or 'gpu'.
                update_state: bool. if True, update state_init from data_train or jnp.ones.
                data_train: DataInput. use for better initialization of state_init.
        '''

        # step 0: save info
        self.model_info = model_info   
        self.train_device = train_device
        self.test_device = test_device
        
        # step 1: initilize model 
        self.net_train, params_init, state_init, self.ol = init_model(self.model_info, mode='train', initialization=True, batch_dim=batch_dim, update_state=update_state, data_train=data_train)
        self.net_test = init_model(self.model_info, mode='test', initialization=False, batch_dim=False)
        self.apply = jax.jit(partial(self.net_test.apply, mutable='state'), static_argnums=3, backend=self.test_device)
        
        # step 2: initilize optimizer and update function                                                                     
        self.optimizer = optax.experimental.split_real_and_imaginary(optax.adam(learning_rate=1e-4) )    # default optimizer: complex adam with lr = 1e-4
        opt_state_init = self.optimizer.init(params_init)
        self.TS = TrainState(epochs=0, params=params_init, state=state_init, opt_state=opt_state_init,state_init=state_init, l_list=[])
        self.update_step = construct_update(self.net_train, self.optimizer, device=train_device)
        self.dataloader = None
        print('Initialization complete.')


    def update_device(self, train_device:str=None, test_device:str=None):
        '''
        update device.
        '''
        if train_device != None: 
            self.train_device = train_device
            self.update_step = construct_update(self.net_train, self.optimizer, device=train_device)
            print('Train device update complete.')
        
        if test_device != None: 
            self.test_device = test_device
            self.apply = jax.jit(partial(self.net_test.apply, mutable='state'), static_argnums=3, backend=self.test_device)
            print('Test device update complete.')

    
    def update_data(self, data_train:DataInput, data_info:dict, batch_size, count:int=0):
        '''
        update train data.
        '''
        if data_train != None:
            self.dataloader = DL.DataLoader(data_train, data_info, batch_size, self.ol, count=count)
            state = self.TS.state_init
            self.TS = self.TS.replace(state_init=state)
        print('Dataset update complete.')
    

    def update_optimizer(self, optimizer:optax.GradientTransformation=None, device:str='cpu'):
        '''
        update optimizer and update function.
        '''
        self.optimizer = optimizer
        opt_state = self.optimizer.init(self.TS.params)
        self.TS = self.TS.replace(opt_state=opt_state)
        self.update_step = construct_update(self.net_train, self.optimizer, device=device)

        print('Optimizer update complete.')


        
    
    def train(self, iters: int):
        '''
            Train out model.
        '''
        # dataset
        if not hasattr(self, 'dataloader') or self.dataloader == None: 
            print('Please update data first.')
            return

        l_list = []
        params, state, opt_state = self.TS.params, self.TS.state, self.TS.opt_state
        for i in tqdm(range(self.TS.epochs, self.TS.epochs + iters),total=iters, desc='training', leave=False):
            y,x = self.dataloader.get_data()
            params, state, opt_state, l = self.update_step(params, state, opt_state, y, x)
            l_list.append(l.item())   
        
        self.TS = TrainState(epochs=self.TS.epochs + iters,params=params,state=state,opt_state=opt_state,state_init=self.TS.state_init, l_list=self.TS.l_list + l_list)
    
        print('Training Finished!')
    

    def dsp(self, data):
        state_init = jax.tree_map(lambda x:jnp.mean(x,axis=0).astype(x.dtype), self.TS.state_init)
        # state_init = jax.tree_map(lambda x:jnp.mean(x,axis=0), self.TS.state)
        v = {'params':self.TS.params, 'state':state_init}
        y, x = wrap_signal(data)
        update_state = True
        z, state_end = self.apply(v, y, x, update_state)
        return z, x
    

    def test(self, data:DataInput, eval_range=(0.2,1)) -> pandas.DataFrame:
        '''
        Test Model.
        '''
        z,x = self.dsp(data)
        a,b = z.val, x.val[z.t.start: z.t.stop]

        N = x.val.shape[-2]
        i1 = int(N*eval_range[0])
        i2 = int(N*eval_range[1])
        a_ = a[i1:i2,:]
        b_ = b[i1:i2,:]
        return BER(a_,b_)
    

    def save(self, path, save_memory=True):
        if save_memory and self.dataloader != None: ds = self.dataloader.ds
        if save_memory and self.dataloader != None: self.dataloader.remove_data()
        
        with open(path,'wb') as file:
            cloudpickle.dump(self, file)
        
        if save_memory and self.dataloader != None: self.dataloader.ds = ds

        print('Model saved.')
    


    def show_loss(self, window:int=200,label=None):
        from .utils import smooth
        l = smooth(self.TS.l_list, window_size=window)
        plt.plot(l, linewidth=2,label=label)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')

    
    def show_symb(self, z, x, size=10):
        from .utils import show_symb
        sig = z.val
        symb = x.val[z.t.start: x.val.shape[-2] +  z.t.stop]
        show_symb(sig, symb, s=size)

    

def test_model(path, test_data):
    a = Model.load(path)
    Q = [a.test(data, eval_range=(0.2,1))['Qsq']['dim0'] for data in test_data]
    return Q