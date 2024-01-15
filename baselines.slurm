#!/bin/bash
#SBATCH -J test            # Job名
#SBATCH -o ./_out_/test_model.out  # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short        # qos(quality of service): normal or short or debug, 对应不同优先级及最大可用时长.
#SBATCH -p V100            # 指定partition: geforce,k80,etc.
#SBATCH --cpus-per-task=8  # 申请 cpu core 数; 可用内存与申请 cpu core 数成正比.
#SBATCH --mem=80G          # 申请10G内存
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 10:00:00        # 申请Job运行时长0小时5分钟0秒,若要申请超过一天时间,如申请1天,书写格式为#SBATCH -t 1-00:00:00

# 上述 SBATCH 参数不指定时均有系统指定的默认值

# 随着 Job 的提交和执行，slurm 会帮助用户在申请的节点上挨个执行下述命令
module add anaconda           # 载入 anaconda 模块
source ~/.bashrc
conda activate fiber


python -m Jobs.baselines --data_path data/test_data.pkl --method CDC --Q_path Qfactor/largetest_MTLoss/CDC.pkl 

stps="1 2 3 4 5 6 7 8 9 10 20 40"


for i in $stps; do
    echo "DBP stps: $i"
    python -m Jobs.baselines --data_path data/test_data.pkl --method DBP --stps $i --Q_path Qfactor/largetest_MTLoss/DBP_stps$i.pkl 
done

for i in $stps; do
    echo "DBP stps: $i"
    python -m Jobs.baselines --data_path data/test_data_ood.pkl --method DBP --stps $i --Q_path Q1000/ood/DBP_stps$i.pkl   --lead_symbols 1000
done