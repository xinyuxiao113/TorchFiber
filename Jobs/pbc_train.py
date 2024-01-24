"""
Train PBC model on seperated data. (different Rs, Nch)

python -m  Jobs.train_sep_data 
"""
import argparse, yaml
from src.TorchDSP.train_pbc import train_model

parser = argparse.ArgumentParser(description="Traing PBC model on seperated data.")
parser.add_argument('--config', type=str, default='configs/simulation.yaml', help='Path to the YAML configuration file')
_args = parser.parse_args()
with open(_args.config, 'r') as file:
    args = yaml.safe_load(file)

if 'model_path' not in args.keys(): args['model_path'] = _args.config.replace('configs', '_models').replace('.yaml', '')
if 'tensorboard_path' not in args.keys(): args['tensorboard_path'] = _args.config.replace('configs', '_outputs/log_tensorboard').replace('.yaml', '')

train_model(args)