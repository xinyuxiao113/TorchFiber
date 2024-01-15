import pickle, torch, argparse, time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.TorchDSP.nneq import NNeq
from src.TorchDSP.dataloader import opticDataset


parser = argparse.ArgumentParser()
parser.add_argument('--method',   help='method', type=str, default='MLP')
parser.add_argument('--model_path',   help='method', type=str, default='nneq')
parser.add_argument('--batch_size',   help='method', type=int, default=4000)
parser.add_argument('--epochs',   help='epochs', type=int, default=20)
parser.add_argument('--lr',   help='lr', type=float, default=1e-3)
parser.add_argument('--M',   help='Memory of each symbol', type=int, default=101)
parser.add_argument('--Rs',   help='model: Rs', type=int, default=40)
parser.add_argument('--Nch',   help='Nch', type=int, default=1)
parser.add_argument('--res_net',   help='Nch', type=bool, default=False)
parser.add_argument('--power_fix',   type=int, default=0)
parser.add_argument('--loss_type',   type=str, default='MTLoss')
args = parser.parse_args()


# Dataset
Nmodes = 1
Nch = args.Nch
Rs = args.Rs
M = args.M
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
train_dataset = opticDataset(Nch, Rs, M, path='data/train_data_afterCDCDSP.pkl', power_fix=args.power_fix)
test_dataset = opticDataset(Nch, Rs, M, path='data/test_data_afterCDCDSP.pkl', power_fix=args.power_fix)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# loss function
def MTLoss(predict, truth):
    return torch.mean(torch.log(torch.mean(torch.abs(predict - truth)**2, dim=-1)))  # predict, truth: [B, Nmodes]

def MeanLoss(predict, truth):
    return torch.mean(torch.abs(predict - truth)**2)  # predict, truth: [B, Nmodes]

criterion = MeanLoss if args.loss_type == 'MeanLoss' else MTLoss

# use train_loader
net = NNeq(M, Nmodes=1, method=args.method, res_net=args.res_net)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)

checkpoint0 = {'model': net.state_dict(),
                    'optimizer': None,
                    'loss': None,
                    'epoch': None,
                    'batch_size': batch_size,
                    'Nch':Nch,
                    'Rs':Rs,
                    'model info': {'M':M, 'Nmodes':Nmodes, 'method':args.method, 'res_net':args.res_net}
                    }
torch.save(checkpoint0, args.model_path + f'_{0}.pth')

if args.epochs > 0:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    Test_l = []
    Train_l = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (y, x, info) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = net(y)
            loss = criterion(y_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % 50 == 0: print(f'Epoch {epoch+1}  batch {batch_idx}  batch Loss={loss.item()}')

        t1 = time.time()
        print(f'Epoch {epoch+1}  Time {t1-t0}')

        trl = epoch_loss/(batch_idx+1) # type:ignore
        # tel = criterion(net(torch.from_numpy(ytest).to(device)), torch.from_numpy(xtest).to(device))
        print(f'Epoch {epoch+1} Train Loss {trl}')
        Train_l.append(trl)
        
        # Test_l.append(tel.item())
    

        checkpoint = {'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': Train_l,
                        'epoch': epoch,
                        'batch_size': batch_size,
                        'Nch':Nch,
                        'Rs':Rs,
                        'model info': {'M':M, 'Nmodes':Nmodes, 'method':args.method, 'res_net':args.res_net}
                        }
        torch.save(checkpoint, args.model_path + f'_{epoch+1}.pth')