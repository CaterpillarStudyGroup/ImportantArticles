# A COMPREHENSIVE ANALYSIS OF PINNS: VARIANTS, APPLICATIONS, AND CHALLENGES

物理信息神经网络（PINN）作为经典神经网络的新型变体，专为求解偏微分方程及其衍生形式而开发。与传统数值方法相比，PINN具有以下优点：  
- 采用无网格化方法，能够有效处理具有不规则、复杂或高维几何特征的问题
- 具有理解并编码物理先验知识的能力，从而生成有效近似解
- 能够从未标注的训练数据中自主推导规律

|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
|180|2019|Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations|PINN|求解偏微分方程及其衍生形式|[link](https://neuralfields.cs.brown.edu/paper_4.html)|


## PINNs Architecture

输入：指定积分域内的坐标点  
输出：对应微分方程的近似解

[TODO] 图1

**3.3 损失函数构建**

考虑参数化偏微分方程的一般表达式：
$$
\begin{aligned}
\text{偏微分方程：} f\left(x, t, \frac{\partial y}{\partial x}, \frac{\partial y}{\partial t}, \ldots; \Psi\right) &= 0, \quad x \in \Omega, \; t \in [0, T] \\
\text{初始条件IC：} y(x, t_0) &= h(x), \quad x \in \Omega \\
\text{边界条件BC：} y(x, t) &= g(t), \quad x \in \partial\Omega, \; t \in [0, T]
\end{aligned}
$$

该方程定义在域 \(\Omega \subset \mathbb{R}^N\) 上，边界为 \(\partial\Omega\)。其中：  
- \(x = (x_1, x_2, \cdots, x_N) \in \mathbb{R}^N\) 表示空间坐标，
- \(t\) 表示时间，
- \(f\) 是描述问题的函数，包含微分算子及参数 \(\Psi\)。
- \(y(x, t)\) 是偏微分方程的解，
- 初始条件为 \(h(x)\)，
- 边界条件为 \(g(t)\)（可以是狄利克雷、诺伊曼、罗宾或周期性边界条件）。

利用神经网络的通用逼近能力，可以构建 \(y(x, t)\) 的代理解 \(\hat{y}(x, t; \theta)\)，其中 \(\theta\) 表示神经网络中的权重和偏置向量集合：

$$
y(x, t) \approx \hat{y}(x, t; \theta)
$$

损失函数定义为：

$$
\begin{aligned}
\mathcal{L}(\Theta) &= w_f \mathcal{L}_f(\theta) + w_{ic} \mathcal{L}_{ic}(\theta) + w_{bc} \mathcal{L}_{bc}(\theta) \\
\mathcal{L}_f(\theta) &= \frac{1}{N_f} \sum_{i=1}^{N_f} \left\| f\left(x, t, \frac{\partial \hat{y}}{\partial x}, \frac{\partial \hat{y}}{\partial t}, \ldots; \Psi\right) \right\|_2^2 \\
\mathcal{L}_{ic}(\theta) &= \frac{1}{N_{ic}} \sum_{i=1}^{N_{ic}} \left\| \hat{y}(x, t_0) - h(x) \right\|_2^2 \\
\mathcal{L}_{bc}(\theta) &= \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left\| \hat{y}(x, t) - g(t) \right\|_2^2
\end{aligned}
$$

其中：  
- \(N_f\) 是配置点集合，
- \(N_{ic}\) 是满足初始条件的点集合，
- \(N_{bc}\) 是满足边界条件的点集合，
- \(w_f\)、\(w_{ic}\) 和 \(w_{bc}\) 是相应的权重系数。

## PINN解ODE

相较于传统数学方法，基于深度学习的方法在求解ODE时展现出多方面的显著优势:  
1. 无论求解过程涉及的数学方法多么复杂，这类方法生成的解都具有较高的精确度。
2. 边界条件与维度因素是制约数学方法效能的关键要素，而深度学习方法对这两个因素均具备良好的适应性。
3. 对于具有随机分布或噪声的数据，此类方法也能有效求解。

当前，用于求解ODE的主流深度学习技术有神经ODE、物理信息神经网络、生成对抗网络。本文专注于第二种。  

[TODO] 表2

|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
||2023|Solving stiff ordinary differential equations using physics informed neural networks (pinns)|用PINN求解刚性ODE|
||2023|Solving differential equations using physics informed deep learning: a hand-on tutorial with benchmark tests.|系统阐述用于求解ODE的DL技术从传统NN到PINN的演变历程。|1. 详细解释了设计PINN涉及的多种因素，包括损失函数构建、物理概念的作用以及优化方法等。<br>2. 该网络在不同ODE上进行了性能测试，并与经典积分方法进行了对比验证。<br>作者发现，PINN的主要优势在于：<br>对于弱非线性问题，仅需后者（传统方法）数据量的一小部分，即可产生与当前任何常用技术相媲美的结果。<br>对于高度非线性问题，PINN在常规条件下难以取得良好效果，需要在一定的积分区间内获得训练数据的先验知识以弥补性能不足。|
||2021|Solving ordinary differential equations using an optimization technique based on training improved artificial neural networks|使用基于DL求解ODE，可被视为推动PINN发展的关键因素之一。|提出了一种借助改进型ANN识别ODE数值解的新方法：<br>1. 先计算特定ODE的近似解，再进行损失最小化。<br>2. 损失函数由多个误差计算函数组合而成。<br>3. 网络参数基于Levenberg-Marquardt算法的结果进行了重构。<br>所提网络能实现更高的精度和更快的收敛速度。|
||2020|A tutorial on solving ordinary differential equations using python and hybrid physics-informed neural network. |使用PINN求解ODE的研究仍处于较浅层面，未能形成系统性的发现。|首次对PINN在ODE求解中的应用进行了较为全面的探讨。该文献着重从实现角度出发，基于经典Python框架进行技术阐释。但并未过度聚焦物理概念本身，而是将数据驱动核作为一种更便捷的模型训练收敛途径。因此，所构建的混合网络同时融合了物理概念与数据驱动核的双重特性。|[link](https://www.sciencedirect.com/science/article/pii/S095219762030292X)|


## PINN解PDE

PDE至今仍无法高效生成解析解。目前已有多种成熟的数值方法复杂度较高。

[TODO] 表3

## PINN解分数阶微分方程(FDE)

[TODO] 表4

## PINN变种

## PINN应用

### 流体力学

该领域大部分问题可归结为NS方程组的求解范畴，而这组方程恰恰适合通过PINN模型进行有效逼近。

相较于传统数值方法，PINNs在流体力学应用中的核心优势在于：  
1. 同一模型能同时处理正问题与反问题。
2. PINNs能有效融合流动观测数据与物理控制方程，实现数据与物理机理的双重驱动。

