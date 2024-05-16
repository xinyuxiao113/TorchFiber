'''
    Meta optimizers.
'''
import torch, torch.nn as nn
from .layers import ComplexLSTM, ComplexLinear, CReLU, MLP, ComplexGRU
from .utils import flat_pytree, unflat_pytree, tree_map, pre_transform


def complex_sigmoid(x):
    return torch.sigmoid(x.real + x.imag)


class MetaNone(nn.Module):
    '''
    Gradient descent.  Not learnable.
    
    Attributes:
        lr_init: learning rate of (w, f) in ADFCell.  w: filter weight. f: phase rotation.

    '''
    def __init__(self, lr_init=(1/2**6, 1/2**7)):
        super(MetaNone, self).__init__()
        self.lr_init = lr_init

    def forward(self, opt_state, grads, params):
        ''' 
        opt_state, grads, params -> opt_state, updates
        '''
        updates = ()
        for grad, lr in zip(grads, self.lr_init):
            updates = updates + (-grad * lr,)
        return opt_state, updates, None

    def init_carry(self, params):
        '''
        Initialize carry state about MetaOpt.
        '''
        return tuple([torch.zeros_like(params[i]) for i in range(len(params))]) 
    

class MetaLr(nn.Module):
    '''
    Gradient descent: learning the lr about ADF.

    Attributes:
        lr_init: initial learning rate of (w, f) in ADFCell.  w: filter weight. f: phase rotation.
    '''
    def __init__(self, lr_init=(1/2**6, 1/2**7)):
        super(MetaLr, self).__init__()
        self.lr = nn.Parameter(torch.tensor(lr_init, dtype=torch.float32), requires_grad=True)

    def forward(self, opt_state, grads, params):
        ''' 
        opt_state, grads, params -> opt_state, updates
        '''
        updates = ()
        for i, grad in enumerate(grads):
            updates = updates + (-grad * self.lr[i],)
        return opt_state, updates, None

    def init_carry(self, params):
        return tuple([torch.zeros_like(params[i]) for i in range(len(params))])
    

class NLMSOpt(nn.Module):
    '''
    Normalized Least Mean Square. Not learnable.
    
    Attributes:
        lr_init: initial learning rate of (w, f) in ADFCell.  w: filter weight. f: phase rotation.
    '''
    def __init__(self, lr_init=(1/2**6, 1/2**7)):
        super(NLMSOpt, self).__init__()
        self.lr = 1e-2
        self.gamma0 = 0.1
        self.eps = 1e-6

    def forward(self, opt_state, grads, o):
        ''' 
        opt_state, grads, o -> opt_state, updates
        '''
        g, tree, shapes = flat_pytree(grads)
        o, _, _ = flat_pytree(o)
        opt_state = self.gamma0 * opt_state + (1 - self.gamma0) * torch.abs(o)**2
        updates =  - self.lr * g / (opt_state + self.eps)
        updates = unflat_pytree(updates, tree, shapes)
        return opt_state, updates, None

    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)
        return torch.zeros_like(I, device=device, dtype=torch.float32)


class RMSPropOpt(nn.Module):
    '''
    Rooted Mean Square. Not learnable. https://paperswithcode.com/method/rmsprop
    
    Attributes:
        lr_init: initial learning rate of (w, f) in ADFCell.  w: filter weight. f: phase rotation.
    '''
    def __init__(self, lr_init=(1/2**6, 1/2**7)):
        super(RMSPropOpt, self).__init__()
        self.lr = 1e-3
        self.gamma0 = 0.1
        self.eps = 1e-6

    def forward(self, opt_state, grads, params):
        ''' 
        opt_state, grads, o -> opt_state, updates
        '''
        g, tree, shapes = flat_pytree(grads)
        opt_state = self.gamma0 * opt_state + (1 - self.gamma0) * torch.abs(g)**2
        updates =  - self.lr * g / (torch.sqrt(opt_state) + self.eps)
        updates = unflat_pytree(updates, tree, shapes)
        return opt_state, updates, None

    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)
        return torch.zeros_like(I, device=device, dtype=torch.float32)



class MetaAdam(nn.Module):   
    '''
        Meta Adam ADF. Learnable Adam.
    '''
    def __init__(self, lr_init=1e-2):
        super(MetaAdam, self).__init__()
        self.lr = nn.Parameter(torch.tensor(1e-3))
        self.beta1 = nn.Parameter(torch.tensor(0.1))
        self.beta2 = nn.Parameter(torch.tensor(0.1))
        self.eps = 1e-8
        
        
    def forward(self, opt_state, grads, params):
        # 获取可学习参数 
        g, tree, shapes = flat_pytree(grads)
        v = (self.beta1*opt_state['v'] + (1-self.beta1)* g) 
        s = (self.beta2*opt_state['s'] + (1-self.beta2)* torch.abs(g)**2) 
        v_hat = v / (1-self.beta1**opt_state['t'])
        s_hat = s / (1-self.beta2**opt_state['t'])
        updates = - self.lr * v_hat / (torch.sqrt(s_hat) + self.eps) 
        updates = unflat_pytree(updates, tree, shapes)
        opt_state = {'v':v, 's':s, 't':opt_state['t']+1}
        return opt_state, updates, None

        
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)
        v = torch.zeros_like(I, device=device)  # [N, 1]
        s = torch.zeros_like(I, device=device, dtype=torch.float32)  # [N, 1]
        return {'v':v, 's':s, 't':1}
    
    

class MetaLSTMOpt(nn.Module):
    '''
        Meta LSTM ADF. Learnable.
        Choose (grads, params) as input.  (N, 2)

        dimension: 2 -> hidden_dim -> hidden_dim -> hidden_dim -> 2*hidden_dim -> 1
                  linear_in       LSTM          LSTM     linear_out_1    linear_out_2            
    ''' 
    def __init__(self, step_max=5e-2, hiddden_dim=16, num_layers=2):
        # control lr in a interval !!
        super(MetaLSTMOpt, self).__init__()
        self.linear_in = nn.Sequential(ComplexLinear(2, hiddden_dim), CReLU())
        self.LSTM = ComplexLSTM(hiddden_dim, hiddden_dim, num_layers)                                   
        self.linear_out = nn.Sequential(ComplexLinear(hiddden_dim,hiddden_dim*2), CReLU(), ComplexLinear(hiddden_dim*2,1))
        self.step_max = step_max

    def forward(self, opt_state, grads, params):
        # step 0: choose info to embed
        I0, tree, shapes = flat_pytree(grads)               # (N,1), N = number of parameters.
        add_info, _, s1 = flat_pytree(params)               # (N,1), N = number of parameters.  assert shapes == s1
        I = torch.concatenate([I0, add_info], dim=-1)       # (N, 2)  complex
        I = pre_transform(I)                                # (N, 2)  complex
        I = self.linear_in(I)                               # (N, 16)  
        I = I.unsqueeze(1)                                  # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        output, opt_state = self.LSTM(I, opt_state)         # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        lr = self.linear_out(output.squeeze(1))             # lr:(N, 1)               
        lr = - complex_sigmoid(lr) * self.step_max          # (N, 1)            
        updates = I0 * lr                                   # (N, 1)
        updates = unflat_pytree(updates, tree, shapes)      # (taps, Nmodes, Nmodes)
        lrs = unflat_pytree(lr, tree, shapes)
        return opt_state, updates, lrs
    
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        return self.LSTM.init_carry(N, device=device)
    


class MetaLSTMtest(nn.Module):
    '''
        Meta LSTMtest ADF. Learnable.
        Choose (grads, params, e, P, Fs, Nch) as input.   (N, 6)

        dimension: 6  -> hidden_dim -> hidden_dim -> 1
                    LSTM          LSTM     linear_out 
    '''
    def __init__(self, step_max=5e-2, hidden_dim=16, num_layers=2):
        # control lr in a interval !!
        super(MetaLSTMtest, self).__init__()
        self.LSTM = ComplexLSTM(6, hidden_dim, num_layers)                                   
        self.linear_out = nn.Sequential( ComplexLinear(hidden_dim,1))
        self.step_max = step_max

    def forward(self, opt_state, grads, params, more_info):    # more_info = (u, d, z, e)
        # step 0: choose info to embed
        grad_info, tree, shapes = flat_pytree(grads)        # (N,1), N = number of parameters.
        param_info, _, _ = flat_pytree(params)              # (N,1), N = number of parameters.  assert shapes == s1
        add_info = self.expand_info(more_info)              # 
        I = [flat_pytree(info)[0] for info in add_info]     # list     
        I = torch.concatenate([grad_info, param_info] + I, dim=-1)      # I: (N, in_dim)     in_dim = 6    
        I = pre_transform(I)                                # (N, in_dim)  complex           in_dim = 6    
        I = I.unsqueeze(1)                                  # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        output, opt_state = self.LSTM(I, opt_state)         # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        lr = self.linear_out(output.squeeze(1))             # lr:(N, 1)               
        lr = - complex_sigmoid(lr) * self.step_max          # (N, 1)            
        updates = grad_info * lr                            # (N, 1)
        updates = unflat_pytree(updates, tree, shapes)      # (taps, Nmodes, Nmodes)
        lrs = unflat_pytree(lr, tree, shapes)
        return opt_state, updates, lrs
    
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        return self.LSTM.init_carry(N, device=device)
    
    def expand_info(self, more_info):
        '''
        more_info = (u, d, z, e, task_info)
        u: [batch, taps, Nmodes]
        d: [batch, Nmodes]
        z: [batch, Nmodes]
        e: ([batch, Nomdes], [batch, Nmodes])  (ew, ef)
        task_info: [batch, task_dim] [P, Fi, Fs, Nch]

        output: dim = 4
        '''
        u, d, z, e, task_info = more_info
        B, D, Nmodes = u.shape
        task_info = task_info / torch.tensor([1, 1.9e14, 8e10, 10]).to(u.device)

        ew = e[0].repeat(Nmodes,D,1,1).permute(2,3,0,1)
        ef = e[1]
        e_info = (ew, ef)

        tw = task_info[:, [0,2,3]].repeat(Nmodes, Nmodes, D, 1,1).permute(3,0,1,2,4)  
        tf = task_info[:, [0,2,3]].repeat(Nmodes, 1,1).permute(1,0,2)
        output = [e_info] + [(tw[...,i], tf[...,i]) for i in range(3)]

        return output


class MetaLSTMplus(nn.Module):
    '''
        Meta LSTMplus ADF. Learnable.
        Choose (grads, params, e, P, Fs, Nch) as input.   (N, 6)

        dimension: 6 -> hidden_dim -> hidden_dim -> hidden_dim -> 2*hidden_dim -> 1
                  linear_in       LSTM          LSTM     linear_out_1    linear_out_2   
    '''
    def __init__(self, step_max=5e-2, hidden_dim=16, num_layers=2):
        # control lr in a interval !!
        super(MetaLSTMplus, self).__init__()
        self.linear_in = nn.Sequential(ComplexLinear(6, hidden_dim), CReLU())
        self.LSTM = ComplexLSTM(hidden_dim, hidden_dim, num_layers)                                   
        self.linear_out = nn.Sequential(ComplexLinear(hidden_dim, 2*hidden_dim), CReLU(), ComplexLinear(2*hidden_dim,1))
        self.step_max = step_max

    def forward(self, opt_state, grads, params, more_info):    # more_info = (u, d, z, e)
        # step 0: choose info to embed
        grad_info, tree, shapes = flat_pytree(grads)        # (N,1), N = number of parameters.
        param_info, _, _ = flat_pytree(params)              # (N,1), N = number of parameters.  assert shapes == s1
        add_info = self.expand_info(more_info)
        I = [flat_pytree(info)[0] for info in add_info]     # list     
        I = torch.concatenate([grad_info, param_info] + I, dim=-1)      # I: (N, in_dim)     in_dim = 6    
        I = pre_transform(I)                                # (N, in_dim)  complex           in_dim = 6    
        I = self.linear_in(I)                               # (N, Hdim) 
        I = I.unsqueeze(1)                                  # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        output, opt_state = self.LSTM(I, opt_state)         # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        lr = self.linear_out(output.squeeze(1))             # lr:(N, 1)               
        lr = - complex_sigmoid(lr) * self.step_max          # (N, 1)            
        updates = grad_info * lr                            # (N, 1)
        updates = unflat_pytree(updates, tree, shapes)      # (taps, Nmodes, Nmodes)
        lrs = unflat_pytree(lr, tree, shapes)
        return opt_state, updates, lrs
    
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        return self.LSTM.init_carry(N, device=device)
    
    def expand_info(self, more_info):
        '''
        more_info = (u, d, z, e, task_info)
        u: [batch, taps, Nmodes]
        d: [batch, Nmodes]
        z: [batch, Nmodes]
        e: ([batch, Nomdes], [batch, Nmodes])  (ew, ef)
        task_info: [batch, task_dim] [P, Fi, Fs, Nch]

        output: dim = 4
        '''
        u, d, z, e, task_info = more_info
        B, D, Nmodes = u.shape
        task_info = task_info / torch.tensor([1, 1.9e14, 8e10, 10]).to(u.device)  # [batch, task_dim]

        ew = e[0].repeat(Nmodes,D,1,1).permute(2,3,0,1)  # [batch, Nmodes, Nmodes, D]
        ef = e[1]                                        # [batch, Nmodes]
        e_info = (ew, ef)                                # ([batch, Nmodes, Nmodes, D], [batch, Nmodes])

        tw = task_info[:, [0,2,3]].repeat(Nmodes, Nmodes, D, 1,1).permute(3,0,1,2,4)  
        tf = task_info[:, [0,2,3]].repeat(Nmodes, 1,1).permute(1,0,2)
        output = [e_info] + [(tw[...,i], tf[...,i]) for i in range(3)]

        return output


class MetaGRUOpt(nn.Module):
    '''
        Meta GRU ADF. Learnable.
        Choose (grads, params, e, P, Fs, Nch) as input.   (N, 6)

        dimension: 6 -> hidden_dim -> hidden_dim -> hidden_dim -> 1
                  linear_in       LSTM          LSTM     linear_out
    '''
    def __init__(self, step_max=5e-2, hidden_dim=16, num_layers=2):
        super(MetaGRUOpt, self).__init__()
        self.LSTM = ComplexGRU(hidden_dim, hidden_dim, num_layers)    
        self.linear_in = nn.Sequential(ComplexLinear(6, hidden_dim), CReLU())                           
        self.linear_out = nn.Sequential( ComplexLinear(hidden_dim,1))
        self.step_max = step_max

    def forward(self, opt_state, grads, params, more_info):    # more_info = (u, d, z, e)
        # step 0: choose info to embed
        grad_info, tree, shapes = flat_pytree(grads)        # (N,1), N = number of parameters.
        param_info, _, _ = flat_pytree(params)              # (N,1), N = number of parameters.  assert shapes == s1
        add_info = self.expand_info(more_info)
        I = [flat_pytree(info)[0] for info in add_info]     # list     
        I = torch.concatenate([grad_info, param_info] + I, dim=-1)      # I: (N, in_dim)     in_dim = 6    
        I = pre_transform(I)                                # (N, in_dim)  complex           in_dim = 6  
        I = self.linear_in(I)                               # (N, in_dim)  complex           in_dim = 6  
        I = I.unsqueeze(1)                                  # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        output, opt_state = self.LSTM(I, opt_state)         # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        lr = self.linear_out(output.squeeze(1))             # lr:(N, 1)               
        lr = - complex_sigmoid(lr) * self.step_max          # (N, 1)            
        updates = grad_info * lr                            # (N, 1)
        updates = unflat_pytree(updates, tree, shapes)      # (taps, Nmodes, Nmodes)
        lrs = unflat_pytree(lr, tree, shapes)
        return opt_state, updates, lrs
    
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        return self.LSTM.init_carry(N, device=device)
    
    def expand_info(self, more_info):
        '''
        more_info = (u, d, z, e, task_info)
        u: [batch, taps, Nmodes]
        d: [batch, Nmodes]
        z: [batch, Nmodes]
        e: ([batch, Nomdes], [batch, Nmodes])  (ew, ef)
        task_info: [batch, task_dim] [P, Fi, Fs, Nch]

        output: dim = 4
        '''
        u, d, z, e, task_info = more_info
        B, D, Nmodes = u.shape
        task_info = task_info / torch.tensor([1, 1.9e14, 8e10, 10]).to(u.device)

        ew = e[0].repeat(Nmodes,D,1,1).permute(2,3,0,1)
        ef = e[1]
        e_info = (ew, ef)

        tw = task_info[:, [0,2,3]].repeat(Nmodes, Nmodes, D, 1,1).permute(3,0,1,2,4)  
        tf = task_info[:, [0,2,3]].repeat(Nmodes, 1,1).permute(1,0,2)
        output = [e_info] + [(tw[...,i], tf[...,i]) for i in range(3)]

        return output
    


class MetaGRUtest(nn.Module):
    '''
        Meta GRUtest ADF. Learnable.
        Choose (grads, params) as input.   (N, 2)

        dimension: 2 -> hidden_dim -> hidden_dim -> hidden_dim -> 1
                  linear_in       LSTM          LSTM     linear_out
    '''
    def __init__(self, step_max=5e-2, hidden_dim=16, num_layers=2):
        # control lr in a interval !!
        super(MetaGRUtest, self).__init__()
        self.LSTM = ComplexGRU(hidden_dim, hidden_dim, num_layers)    
        self.linear_in = nn.Sequential(ComplexLinear(2,hidden_dim), CReLU())                           
        self.linear_out = nn.Sequential( ComplexLinear(hidden_dim,1))
        self.step_max = step_max

    def forward(self, opt_state, grads, params, more_info):    # more_info = (u, d, z, e)
        # step 0: choose info to embed
        I0, tree, shapes = flat_pytree(grads)               # (N,1), N = number of parameters.
        add_info, _, s1 = flat_pytree(params)               # (N,1), N = number of parameters.  assert shapes == s1
        I = torch.concatenate([I0, add_info], dim=-1)       # (N, 2)  complex
        I = pre_transform(I)                                # (N, 2)  complex
        I = self.linear_in(I)                               # (N, in_dim)  complex           in_dim = 6  
        I = I.unsqueeze(1)                                  # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        output, opt_state = self.LSTM(I, opt_state)         # I: (N, 1, 16)   opt_state: ([depth, N, 16*2], [depth, batch, 16*2])
        lr = self.linear_out(output.squeeze(1))             # lr:(N, 1)               
        lr = - complex_sigmoid(lr) * self.step_max          # (N, 1)            
        updates = I0 * lr                                   # (N, 1)
        updates = unflat_pytree(updates, tree, shapes)      # (taps, Nmodes, Nmodes)
        lrs = unflat_pytree(lr, tree, shapes)
        return opt_state, updates, lrs
    
    def init_carry(self, params, device='cpu'):
        I, tree, shapes = flat_pytree(params)               # (N,1), N = number of parameters.
        N = I.shape[0]
        return self.LSTM.init_carry(N, device=device)
    

# MetaOpt 修改的思路
'''
1. 哪些东西应当作为输入预测 lr ?
2. LSTM: element-wise or vector-wise?

'''

if __name__ == '__main__':
    dtype = torch.complex64
    net = MetaLSTMOpt()
    params = (torch.randn(32, dtype=dtype), torch.randn((1), dtype=dtype))
    grads = (torch.randn(32, dtype=dtype), torch.randn((1), dtype=dtype))
    opt_state = net.init_carry(grads)
    y = net(opt_state, grads, params)
    print(y)
