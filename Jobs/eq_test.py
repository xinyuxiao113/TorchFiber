import os, time, torch, numpy as np, argparse
from torch.utils.tensorboard.writer import SummaryWriter
from src.TorchDSP.dataloader import MyDataset 
from src.TorchDSP.eq import eqAMPBC, eqPBC, eqCNNBiLSTM, eqAMPBCaddNN
from src.TorchDSP.eq import Test, write_log
from src.TorchDSP.pbc import AmFoPBC, TorchSignal, TorchTime
from src.TorchDSP.loss import MSE, Qsq
from src.TorchSimulation.receiver import BER

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, default='_outputs/log_test/NN')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--window_size', type=int, default=41)
parser.add_argument('--Nwindow', type=int, default=400000)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--lr', type=float, default=1e-2)  
parser.add_argument('--seed', type=int, default=1234)

parser.add_argument('--Nch', type=int, default=3)
parser.add_argument('--Rs', type=int, default=40)
parser.add_argument('--Pch', type=int, default=2)

parser.add_argument('--model', type=str, default='CNNBiLSTM')
parser.add_argument('--channels', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=21)
parser.add_argument('--hidden_size', type=int, default=10)

args = parser.parse_args()
os.makedirs(args.log_file, exist_ok=True)
writer = SummaryWriter(args.log_file)

window_size = args.window_size
Nwindow = args.Nwindow
epochs = args.epochs
torch.manual_seed(args.seed)

train_data = MyDataset('dataset_A800/train.h5', Nch=[args.Nch], Rs=[args.Rs], Pch=[args.Pch], 
                       window_size=window_size, strides=1, Nwindow=Nwindow, truncate=20000, 
                       Tx_window=False, pre_transform='Rx_CDCDDLMS(taps=32,lr=[0.015625, 0.0078125])')

test_data = MyDataset('dataset_A800/test.h5', Nch=[args.Nch], Rs=[args.Rs], Pch=[args.Pch], 
                       window_size=window_size, strides=1, Nwindow=20000, truncate=20000, 
                       Tx_window=False, pre_transform='Rx_CDCDDLMS(taps=32,lr=[0.015625, 0.0078125])')

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, drop_last=True)

print('Train Data number:',len(train_data))
print('Test Data number:',len(test_data))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=True, drop_last=True)

if args.model == 'CNNBiLSTM':
    net = eqCNNBiLSTM(M=window_size, channels=args.channels, kernel_size=args.kernel_size , hidden_size=args.hidden_size)
elif args.model == 'AMPBC':
    net = eqAMPBC(M=window_size)
elif args.model == 'AMPBCaddNN':
    net = eqAMPBCaddNN(pbc_info={'M':window_size}, nn_info={'M': window_size, 'channels': args.channels, 'kernel_size': args.kernel_size, 'hidden_size': args.hidden_size})
net.cuda()

if args.model == 'AMPBC' or args.model == 'CNNBiLSTM':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.Adam([{'params': net.pbc.parameters(), 'lr': 1e-5}, {'params': net.nn.parameters(), 'lr': 1e-3}])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)


for epoch in range(epochs):
    train_loss = 0
    for i, (Rx, Tx, info) in enumerate(train_loader):
        Rx, Tx, info = Rx.cuda(), Tx.cuda(), info.cuda()
        y = net(Rx, info)
        loss = MSE(y, Tx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  
        writer.add_scalar('Loss/train_batch', loss.item(), epoch*len(train_loader)+i)
    scheduler.step()
    metric = Test(net, test_loader)
    write_log(writer, epoch, train_loss/len(train_loader), metric)

writer.add_text('Model', str(net))

# # 记录检查点，下次可以从检查点读取模型和优化器继续训练
# ckpt = {
#     'epoch': epoch,
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
# }