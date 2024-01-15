"""
Save predict signal for BER convergence figure.

python -m Jobs.Q_convergnce
"""
import torch, matplotlib.pyplot as plt, time, pickle
import argparse
from src.TorchDSP.dsp import DSP
from src.TorchDSP.dataloader import signal_dataset, get_data
from src.TorchSimulation.receiver import  BER 


test_data, info = get_data('data/test_data_few.pkl')
test_dataset = signal_dataset(test_data, batch_size=360, shuffle=False)

def test_model(dic, dataset, device='cuda:0'):
    model = DSP(**dic['model info'])
    model.load_state_dict(dic['model'])
    model.change_mode('test', batch_size=dataset.batch_size)
    model.eval()

    res = []

    for data in dataset:
        t0 = time.time()

        with torch.no_grad():
            model.adf.init_state()
            model = model.to(device)
            data = data.to(device)
            predict = model(data.signal_input, data.task_info, data.signal_output)
            truth = data.signal_output.val[:, predict.t.start:predict.t.stop]
            model.adf.detach_state()

            res.append((predict.val.to('cpu'), truth.to('cpu'), data.task_info.to('cpu')))
        
        t1 = time.time()
        print('time: ', t1-t0)
    
    return  torch.cat([k[0] for k in res], dim=0), torch.cat([k[1] for k in res], dim=0), torch.cat([k[2] for k in res], dim=0) # type: ignore


paths = ['models/torch_MTLoss/MetaDBP_ddlms_9.pth', 'models/torch_MTLoss/MetaDBP_metagrutest_tiny_old_9.pth']
predict = {}
truth = {}
for path in paths:
    dic = torch.load(path)
    predict[path], truth[path], info = test_model(dic, test_dataset)

print('complete')

import pickle 
pickle.dump((predict, truth, info), open('data/predict_truth_info.pkl', 'wb'))