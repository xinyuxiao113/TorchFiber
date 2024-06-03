
file="dataset_A800/test.h5"

# python -m Jobs.torch_simulation --path $file --config configs/simulation/torch_train.yaml

# python -m Jobs.torch_Rx --path $file 

# python -m Jobs.torch_baselines --path $file --comp CDC 

python -m Jobs.torch_baselines --path $file --comp DBP --stps 1

python -m Jobs.torch_baselines --path $file --comp DBP --stps 2

python -m Jobs.torch_baselines --path $file --comp DBP --stps 4

python -m Jobs.torch_baselines --path $file --comp DBP --stps 8

python -m Jobs.torch_baselines --path $file --comp DBP --stps 16

python -m Jobs.torch_baselines --path $file --comp DBP --stps 32