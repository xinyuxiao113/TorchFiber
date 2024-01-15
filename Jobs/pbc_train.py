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

train_model(args)