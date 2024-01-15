"""
Test NN equalizer.

python -m Jobs.nneq_test --batch_size 4028  --data_path data/train_data_afterCDCDSP.pkl --loss_type MeanLoss
"""
from src.TorchDSP.dataloader import opticDataset, get_k_batch
from torch.utils.data import Dataset
import pickle, torch, argparse
from src.TorchDSP.nneq import NNeq
from src.TorchSimulation.receiver import  BER 


parser = argparse.ArgumentParser()
parser.add_argument('--model_path',   help='method', type=str, default='models/nneq_MTLoss/MLP_19.pth')
parser.add_argument('--data_path',   help='method', type=str, default='data/test_data_afterCDCDSP.pkl')
parser.add_argument('--Q_path',   help='method', type=str, default='Qfactor/nneq_MTLoss/MLP_19.pth')
parser.add_argument('--batch_size',   help='method', type=int, default=10000)
parser.add_argument('--Rs',   help='model: Rs', type=int, default=40)
parser.add_argument('--Nch',   help='Nch', type=int, default=1)
parser.add_argument('--M',   help='Memory of each symbol', type=int, default=101)
parser.add_argument('--power_fix',   type=int, default=0)
parser.add_argument('--loss_type',   type=str, default='MTLoss')
args = parser.parse_args()

# loss function
def MTLoss(predict, truth):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=-1)))  # predict, truth: [B, Nmodes]

def MeanLoss(predict, truth):
    return torch.mean(torch.abs(predict - truth)**2)  # predict, truth: [B, Nmodes]

criterion = MeanLoss if args.loss_type == 'MeanLoss' else MTLoss


# Dataset
Nch = args.Nch
Rs = args.Rs
batch_size = args.batch_size
M = args.M
Nmodes = 1
lr = 1e-3


test_dataset = opticDataset(Nch, Rs, M, path=args.data_path, power_fix=args.power_fix)

# Dataloader
from torch.utils.data import DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

y_ = []
x_ = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dic = torch.load(args.model_path)
net = NNeq(**dic['model info'])
net.load_state_dict(dic['model'])
net.eval()
net = net.to(device)

for batch_idx, (y, x, info) in enumerate(test_dataloader):
    x = x.to(device)
    y = y.to(device)
    y_hat = net(y)
    x_.append(x.cpu().detach())
    y_.append(y_hat.cpu().detach())

predict = torch.cat(y_, dim=0).reshape(test_dataset.y.shape[0], -1, Nmodes)  # [B, M, Nmodes]
truth = torch.cat(x_, dim=0).reshape(test_dataset.y.shape[0], -1, Nmodes)   # [B, M, Nmodes]
test_loss = criterion(predict, truth)
print('Test loss:', test_loss.item())

device = 'cpu'
res = {}
for n in [0, 1000, 10000, 20000]:
    metric = BER(predict[:,n:].to(device), truth[:,n:].to(device))
    res[f'BER from {n}th symb'] = {args.model_path: metric}

pickle.dump((res, test_dataset.task_info), open(args.Q_path, 'wb'))