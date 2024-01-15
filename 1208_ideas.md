
# 1.数据准备
## 1.1 生成网格更密，长度更长的数据
- Nbits: [4e5]
- Power(dBm): [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8]
- Nch:   [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
- Rs:    [10G,20G,30G,40G,50G,60G,70G,80G,90G,100G, 110G, 120G, 130G, 140G, 150G, 160G, 170G, 180G, 190G]
- spacing_factor: [1.2]


## 1.2 训练数据分布
- Power: [-1,0,1]
- Nch:   [5,7]
- Rs:    [40,80]
- spacing_factor: [1.2]

## 1.3 测试数据分布
- Power(dBm): [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
- Nch:   [1, 3, 5, 7, 9, 11]
- Rs:    [20G,40G,80G,160G]
- spacing_factor: [1.2, 1.5]



# 2.模型结构
- Meta-DBP: 设计非线性算子的滤波结构
  - [x] 线性算子：频域 or 时域
    $$
        u_n = IFFT\left(FFT(x_n) \exp\left(-\frac{i\beta_2 h^2}{2}\omega^2\right)\right)
    $$
  - [x] 非线性算子：基于Perturbation的非线性滤波结构，可以编码Rs,Power信息.
    $$
        u_n = x_n\exp\left(i\gamma L_{eff}(h) P^{3/2}\cdot \sum_{j=-k}^{k}c_j|x_{n+j}|^2\right)
    $$
- Meta-MIMO
  - [x] Trainable step.
    $$
        y_n = h_n * x_n \\
        h_{n+1} = U(h_n, x_n, l)
    $$
  - [x] **Auto step**
    $$
    (l_{n+1}, h_{n+1}) = f_{\theta}(x_n, h_n, l_{n})
    $$
  - [x] MLP step.
    $$
        h_{n+1} = f(h_n, x_n, l_{n}),\\
        l_{n+1} = g_{\theta}(h_n, x_n, l_n)
    $$
  - [x] **Multi-scale XPM adaptive filte**r.
    $$
        y_n = \sum_{i=1}^{K} h^{(i)}*x_n,\\
        h^{(i)}_{n+1} \leftarrow U_i(x_n, h^{(i)}_n, l_i)
    $$

detail: 如果要不同Rs,Nch,Power的数据采用同一个网络结构，我们可以将两个算子中的卷积核长度都取得足够长.

# 3.优化方法
- 如何投喂不同模态的数据: Multi-task learning. Meta-learning.
- 如何优化DBP+MIMO的结构: 分步优化, 交替优化方法.


# 4.数值实验计划
预期展示的方法有以下组合，并且展示其在不同数据维度的泛化鲁棒性能.
- CDC + MIMO
- DBP + MIMO
- Meta-DBP + MIMO
- CDC + Meta-MIMO
- DBP + Meta-MIMO
- Meta-DBP + Meta-MIMO

除此之外还可以探讨另外的几个关键因素的改变对实验结果的影响:
- 是否知道准确的光纤参数. (主要考察Meta-DBP能否在模糊光纤参数的情况下从数据中学到物理知识)
- 光纤WDM的通道间隔. (主要影响XPM的性质,考验MIMO的鲁棒性)
- 接收机处的每符号采样数. (待定：类似于FNO的实验，考察模型对于不同分辨率的适应性)



<!-- 1. MetaDSP 消融实验  **
2. PBC + NN equalizer 实验
3. 选课，云南会议  ** -->


论文建议：；
0. 消融实验 (huawei)
1. 测试泛化能力 (huawei)
2. 展示收敛和跟踪能力的区别。 （可以修改LSTM的结构，或者添加惩罚项将跟踪能力提高）
3. Multi-task loss.
4. 设计Meta-ADF的连接结构
5. Meta-ADF:
- 修改训练参数：tbpl, batch, epochs


一些思考：
1. sin activation + MLP(siren) loss function只用函数值。 结果能够保存f梯度信息。能不能做理论分析和推广？

2. PINN loss边界loss,内部loss平衡.

3. Ensemble KF. 一定可以拟合非线性.


10-22 Task
1. 复杂光纤系统模拟

2. NNeq baselines
- CNN+BiLSTM

3. 高阶PBC




# 5. 1208 一些思考



## Theory

Under NLSE, 
$$
u_z = -\frac{\alpha}{2}u-\frac{i\beta_2}{2} u_{tt} - i\gamma |u|^2u
$$
if we set
$$
u(z,t) = \sum_{i=1}^{n} a_i(z) g_i(z, t)
$$
Then we can deduce:
$$
da_p(z)/dz = -i\gamma\sum_{m,n,k} X_{m,n,k}(z) a_{p+m} a_{p+n} a_{p+k}^*
$$


$$
da_p(z)/dz = -i\gamma\sum_{m} C_m(z) |a_{p+m}|^2 a_{p}
$$

$$
a_p(z) = a_p(0) * \exp(iC*|a|^2) = a_p(0) (1 + iC*|a|^2)
$$

PBC: one step Euler scheme

$$
(a_p(L) - a_p(0))/L  = -i\gamma\sum_{m,n,k} X_{m,n,k}(L) a_{p+m}(0) a_{p+n}(0) a_{p+k}^*(0)
$$

$$
a_p(L) = a_p(0) - i\gamma L \sum_{m,n,k} X_{m,n,k}(L) a_{p+m}(0) a_{p+n}(0) a_{p+k}^*(0)
$$

FDBP:
$$
m*n=0, m+n=k
$$

$$
a_p(L) = a_p(0) - i\gamma L \sum_{m} 2c_m(L) |a_{p+m}(0)|^2 a_{p}(0)
$$

PBC:
$$
|m|,|n| \leq L/2,  |m*n| \leq \rho*L/2
$$



## 2023.11 huawei meeting
1. 真实label拿不到，半实时更新C, 如何训练PBC系数
2. 高阶PBC  **
3. 非线性，PDL耦合
4. CDC + ADF + NNeq   **           # CNN+BiLSTM,（<50G）
5. PMD,PDL加上模拟.  
6. 收发端PBC.



## 2023.12 huawei meeting and my ideas
1. 数值格式的收敛性验证. 3,4 steps提升,
2. (?)不同的采样函数带来的结果如何？(采样定理对于别的函数是什么形式)       
3. (?)不同Nch, Rs设定下的性能极限
4. baselines 1: 经典高阶PBC实现，对比  (huawei)
5. baselines 2: PBC+NN. 


6. HPBC学习到的系数能够和数值计算的系数有什么样的对应关系？
7. 将张量做低秩分解，然后PBC格式可以转化为卷积神经网络+Attention.
8. 格式守恒性模型增益.



7. multi-span PBC.  (huawei)
8. 自监督学习,无$a_p(L)$, 加权重忽略一些不太好的信号点 (huawei)
9. 建立直接数值计算得到的系数产生的补偿模型.
10. 迭代修剪PBC系数的策略.


### HoPBC训练的一些idea:
- 使用Adagrad优化器.(实验结果)
- 使用数值计算或者一阶PBC的系数作为初始化.
- 使用张量分解导出的CNN结构.
- 两层PBC的系数index不应该取为一致.

### 训练添加特性：
- 指定特定的功率训练
- 从某个ckpt开始训练
- 使用某种特定的初始化 
- 使用Lr schedule



### 0105 实验安排



