python -m Jobs.torch_simulation --path dataset/train.h5 --config configs/simulation/torch_train.yaml

python -m Jobs.torch_Rx --path dataset/train.h5

python -m Jobs.torch_baselines --path dataset/train.h5 --comp CDC