"""
Test MetaDSP Model. Old version.
"""
import torch, matplotlib.pyplot as plt, time, pickle
import argparse
from src.TorchDSP.dsp import DSP
from src.TorchDSP.dataloader import signal_dataset, get_data
from src.TorchSimulation.receiver import  BER 


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',    help='data path: data/test_data_few.pkl, data/test_data_large.pkl, data/test_data_ood.pkl', type=str, default='data/test_data_few.pkl')
parser.add_argument('--model_path',   help='model path', type=str, default='models/torch/FDBP_ddlms_datafew_9.pth')
parser.add_argument('--Q_path',       help='path to save Q result', type=str, default='Qfactor/MTLoss/FDBP_ddlms_datafew_9.pkl')
parser.add_argument('--lead_symbols', help='lead symbols in ADF: 1000, 2000', type=int, default=2000)
parser.add_argument('--batch_size',   help='Batch size: 120, 360.', type=int, default=240)
parser.add_argument('--BER_n',        help='Calculate BER after discard n symbols.',  type=int,   nargs='+', default=[0, 500, 1000, 1500, 2000, 10000])
args = parser.parse_args()

print('Test-loading data:')
t0 = time.time()
test_data, info = get_data(args.data_path)
dic = torch.load(args.model_path)
test_dataset = signal_dataset(test_data, batch_size=args.batch_size, shuffle=False)
t1 = time.time()
print('time for loading: ', t1-t0, flush=True)


def test_model(dic, dataset: signal_dataset, device='cuda:0'):
    model = DSP(**dic['model info'])
    model.load_state_dict(dic['model'])
    model.change_mode('test', batch_size=dataset.batch_size)
    model.eval()
    model.adf.change_lead_symbols(args.lead_symbols)   

    res = []

    for i,data in enumerate(dataset):
        print(f'Testing batch {i}/{dataset.batch_number()}', flush=True)
        t0 = time.time()
        with torch.no_grad():
            model.adf.init_state(data.signal_input.val.shape[0])
            model = model.to(device)
            data = data.to(device)
            predict = model(data.signal_input, data.task_info, data.signal_output)
            truth = data.signal_output.val[:, predict.t.start:predict.t.stop]
            model.adf.detach_state()
        
        t1 = time.time()
        print('time for batch: ', t1-t0, flush=True)

        res.append((predict.val.to('cpu'), truth.to('cpu'), data.task_info.to('cpu')))
    
    
    return  torch.cat([k[0] for k in res], dim=0), torch.cat([k[1] for k in res], dim=0), torch.cat([k[2] for k in res], dim=0) # type: ignore


predict, truth, task_info = test_model(dic, test_dataset, device='cuda:0')

device = 'cpu'
res = {}

for n in args.BER_n:
    print('Calculate BER discard %d symbols' % n, flush=True)
    t0 = time.time()
    metric = BER(predict[:,n:].to(device), truth[:,n:].to(device))
    res[f'BER from {n}th symb'] = {args.model_path: metric}
    t1 = time.time()
    print('Time for each BER calculation: ', t1-t0, flush=True)

pickle.dump((res, task_info), open(args.Q_path, 'wb'))