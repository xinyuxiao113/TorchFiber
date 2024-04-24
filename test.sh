
file="dataset_exp/train_x4.h5"

# python -m Jobs.torch_simulation --path $file --config configs/simulation/torch_train.yaml

# python -m Jobs.torch_Rx --path $file 

python -m Jobs.torch_baselines --path $file --comp CDC --rx_grp 'Rx(sps=2,chid=0,method=frequency cut)'