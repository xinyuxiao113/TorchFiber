# step 1: simulation dataset (Jax based)
- Transmitter
- channel
```
python -m Jobs.simulation --path your_data_path --batch 2 --Nbits 400000 --SF 1.2 --Nmodes 1 --seed 09071909
```

# step 2: receive data (Jax based)
- receiver
```
python -m Jobs.generate_data
```

# step 3: train model (Torch based)
```
python -m Jobs.Torchtrain --DBP FDBP --ADF ddlms --path your_model_path ...
```

# step 4: test model (Torch based)
```
python -m Jobs.Torchtest  --path your_model_path --name output_name ...
```

# step 5: show Q factor 
use showQ.ipynb



# 模型训练方式：
1.将不同Nch，Rs的数据混合训练，使用同一个模型.
- MetaDSP (Torchtrain.py, Torchtest.py)

2.将不同Nch，Rs的数据分别训练，使用不同的模型.
- PBC (pbc_train.py, pbc_test.py)
- SOPBC (sopbc_train.py, sopbc_test.py)
- HPBC (hpbc_train.py, hpbc_test.py)
- nneq (nneq_train.py, nneq_test.py)