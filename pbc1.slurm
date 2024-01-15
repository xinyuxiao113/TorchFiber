#!/bin/bash
#SBATCH -J test            # Job名
#SBATCH -o ./_outputs/log_slurms/test1.out  # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short        # qos(quality of service): normal or short or debug, 对应不同优先级及最大可用时长.
#SBATCH -p RTX3090         # 指定partition: geforce,k80,etc.
#SBATCH --cpus-per-task=1  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=10G          # 申请10G内存
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 10:00:00        # 申请Job运行时长0小时5分钟0秒,若要申请超过一天时间,如申请1天,书写格式为#SBATCH -t 1-00:00:00

# 上述 SBATCH 参数不指定时均有系统指定的默认值

# 随着 Job 的提交和执行，slurm 会帮助用户在申请的节点上挨个执行下述命令
module add anaconda           # 载入 anaconda 模块
source ~/.bashrc
conda activate fiber

python -m Jobs.pbc_train --config configs/Rs20_Nch1/fopbc_L400/L400_lossMean_Pch-4.yaml
python -m Jobs.pbc_train --config configs/Rs20_Nch1/fopbc_L400/L400_lossMean_Pch-3.yaml
python -m Jobs.pbc_train --config configs/Rs20_Nch1/fopbc_L400/L400_lossMean_Pch-5.yaml
# python -m Jobs.pbc_train --config configs/Rs40/pbcnn_init/L200_Pch-1_H200_init4_bias0.yaml
# python -m Jobs.pbc_train --config configs/Rs40/fopbc_loss/fopbc_L200_Adam_lr1e-5_lossMean.yaml
# python -m Jobs.pbc_train --config configs/Rs40/fopbc_opt/fopbc_L200_SGD_lr5e-6_mmt0.9.yaml
# python -m Jobs.pbc_train --config configs/pbc_Rs40/sopbc_Lk5.yaml
# python -m Jobs.pbc_train --config configs/pbc_Rs40/hopbc_L200_adagrad_lr1e-3.yaml
# python -m Jobs.pbc_train --config configs/pbcnn_Rs80/pbc_nn13.yaml
# python -m Jobs.pbc_train --config configs/hopbc_Rs80/hopbc_L400_Adagrad_lr1e-3_tbpl2000_lossMean.yaml
# python -m Jobs.pbc_train --config configs/hopbc_Rs80/hopbc_L400_Adagrad_lr1e-4_tbpl2000_lossMSE.yaml
# python -m Jobs.pbc_train --config configs/pbc_Rs80/fopbc_L800.yaml
# python -m Jobs.pbc_train --config configs/pbc/pbc_nn.yaml
# python -m Jobs.pbc_train --config configs/pbc/fopbc.yaml
# python -m Jobs.pbc_train --config configs/pbc/pnse.yaml

# L=100    0.5h
# L=200    1h
# L=400    2h
# L=800    4h
