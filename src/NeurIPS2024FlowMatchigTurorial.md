
# NeurIPS 2024 Flow Matchig Turorial


P1     
# Flow Matching Tutorial     

> Flow Matching Basics

P2     
## Agenda   

[40 mins] **01 Flow Matching Basics**     
[35 mins] **02 Flow Matching Advanced Designs**     
[35 mins] **03 Model Adaptation**     
[30 mins] **04 Generator Matching and Discrete Flows**    
[10 mins] **05 Codebase demo**    

P4     
01 Flow Matching Basics   

P6    
WHAT IS FLOW MATCHING?       
A scalable method to train **flow generative models**.      

HOW DOES IT WORK?      
Train by regressing a **velocity**, sample by following the **velocity**      

P8    
## The Generative Modeling Problem

![](assets/P8图.PNG)   

> 正方形代表所有可能的状态所构成的空间，即图像空间正方形中的每个点代表一个sample，即一张图像。      
\\(P\\) 是源分布，\\(Q\\) 是目标分布。     
\\(X_o\\) 和 \\(X_1\\)分别是 \\(P\\) 分布和 \\(Q\\) 分布中的 sample．     
生成模型的目标是，找到一个可以从 \\(P\\) 中 sample 到 \\(Q\\) 中 sample 的映射。    

P9     

> GAN 是一种生成模型，优点是快，因为它的生成过程只需要一个forward.缺点是（1）没有一个精确的可以用于 sample 的概率模型（2）难以训练。     

P10    
## Model

• Continuous-time Markov process       

![](assets/P10图1-1.PNG)

![](assets/P10图2.PNG)


> 增量生成是另一种生成范式，不是直接生成最终结果，而是逐步生成。每一次生成比上一次要好。\\(\Phi\\) 是从一次生成到另一次生成的转移函数。    
Flow 的转移过程是平滑的。Diffusion 是连续但不平滑的。还有一些是不连续的，但都是基于连续时间马尔可夫过程的随机过程。      
生成模型的目标是学习转移函数。      

P11    

## Marginal probability path

![](assets/P11图.PNG)

> 边缘概率路径，是指，任意一个特定的 \\(t\\) 时刻。\\(X_t\\) 所属于的分布 \\(P_t\\). 即连续时间上的分布簇。    
生成模型最重要的是，边缘概率路径以 \\(P\\) 分布开始，以 \\(Q\\) 分布结束。     

P12   
• For now, we focus on flows…    

![](assets/P12图.PNG)

> 流的特点：(1) 确定性，已知 \\(X_t\\)，那么 \\(X_{t+h}\\) 是确定的。(2) 平滑       
流的优势：(1) sample 速度快 (2) 可以构建模型似然的无偏后计器。      
Diffusion 和 Jump 具有更大的设计空间，因此具有更多生成能力。    

P13    
## Flow as a generative model    

![](assets/P13图.PNG)

> \\(\Psi_t\\) 是一个双射函数，因此它可以重塑空间而不丢失信息。    
通过对高维空间的 warping，使 \\(P\\) 分布逐步变为 \\(Q\\) 分布。     











 
