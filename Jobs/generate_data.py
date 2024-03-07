"""
Generate data from received optical signal.

python -m Jobs.generate_data
"""

import src.JaxSimulation.DataLoader as DL 
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file',        help='data file index. Final_batch2 or batch2', type=str, default='/gpfs/share/home/2001110035/data/test')
parser.add_argument('--save_path',   help='save path', type=str, default='../data/t0.pkl')
parser.add_argument('--Nch',         help='Choose number of channels',      type=int,   nargs='+', default=[1])
parser.add_argument('--Rs',          help='Choose single channel symbol rate. [Hz]',  type=int,   nargs='+', default=[10])
parser.add_argument('--Pch',         help='Choose single channel launch power. [dBm]',  type=int,   nargs='+', default=[i for i in range(-8, 9)])
parser.add_argument('--batch_id',    help='Choose the batch id. 0 ',        type=int, nargs='+', default=[0])
args = parser.parse_args()


data, info = DL.Generate_Data(args.file, Nch=args.Nch, Rs=args.Rs, power=args.Pch, batch_id=args.batch_id, merge=True)
pickle.dump((data, info), open(args.save_path, 'wb'))


# file = 'batch2'
# train_data,train_info = DL.Generate_Data(file, Nch=[1,3,5,7,9,11,13], Rs=[20, 40, 60, 80, 100, 120, 140, 160, 180],SF=1.2,mode=1,power=[-8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,7,8], batch_id=[0], merge=True)
# test_data, test_info = DL.Generate_Data(file, Nch=[1,3,5,7,9,11,13], Rs=[20, 40, 60, 80, 100, 120, 140, 160, 180],SF=1.2,mode=1,power=[-8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6,7,8], batch_id=[1], merge=True)


# pickle.dump((train_data, train_info), open('train_data.pkl', 'wb'))
# pickle.dump((test_data, test_info), open('test_data.pkl', 'wb'))


# file = 'Final_batch2'
# train_data,train_info = DL.Generate_Data(file, Nch=[1,3,5,7,9,11], Rs=[20, 40, 80, 160],SF=1.2,mode=1,power=[-8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6], batch_id=[0], merge=True)
# test_data, test_info = DL.Generate_Data(file, Nch=[1,3,5,7,9,11], Rs=[20, 40, 80, 160],SF=1.2,mode=1,power=[-8, -7, -6, -5, -4, -3, -2, -1,0,1,2,3,4,5,6], batch_id=[1], merge=True)


# pickle.dump((train_data, train_info), open('train_data_few.pkl', 'wb'))
# pickle.dump((test_data, test_info), open('test_data_few.pkl', 'wb'))
