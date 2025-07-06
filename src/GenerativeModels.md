# The Generative Modeling Problem

![](./assets/P8图.png)   

> 正方形代表所有可能的状态所构成的空间，即图像空间。正方形中的每个点代表一个sample，即一张图像。      
\\(P\\) 是源分布，\\(Q\\) 是目标分布。     
\\(X_0\\) 和 \\(X_1\\)分别是 \\(P\\) 分布和 \\(Q\\) 分布中的 sample．     
**生成模型的目标是，找到一个可以从 \\(P\\) 中 sample 到 \\(Q\\) 中 sample 的映射**。       

P9     
# 生成模型的范式

> 生成模型有两大类范式：直接生成和增量生成。  

## 直接生成

GAN、VAE 属于第一大类生成模型，优点是快，因为它的生成过程只需要一个forward。  

[TODO] 把下面表格中的图下载下来，换成本地链接

|生成模型|特点|结构|链接|
|---|---|---|---|
|AE|降维、聚类，但latent仍是复杂分布，不能直接sample|![](https://pica.zhimg.com/v2-350331de3d1f8c7c02df2d4a89e1b676_r.jpg)|
|VAE|降维、聚类，latent为std normal，可以直接sample|![](https://pic1.zhimg.com/v2-a9769819dddedc52151bf12f2ac98ad8_1440w.jpg)|
|VQ-VAE|离散AE（用于降维、聚类） + PixelCNN（用于sample）|![](https://pic3.zhimg.com/v2-ece699c485581ba30cc739ce1c51d9b4_1440w.jpg)|
|GAN|

### AE

### VAE

### VQ-VAE

### GAN

GAN的缺点是（1）没有一个精确的可以用于 sample 的概率模型（2）难以训练。     

### 自回归

例如PixelCNN，根据前面的像素生成后面的像素，或者transformer，根据上一帧生成下帧。

## 增量生成 

> 增量生成是另一种生成范式，不是直接生成最终结果，而是逐步生成。每一次生成比上一次要好。

![](./assets/P10图2.png)

|生成模型|特点|链接|
|---|---|---|
|Flow Matching|转移过程是平滑的。| [link](./NeurIPS2024FlowMatchigTurorial/FlowMatchingBasics.md)|
|Diffusion| 转移过程是连续但不平滑的 | [link](./diffusion-tutorial-part/Fundamentals/DenoisingDiffusionProbabilisticModels.md)|
|Jump|转移过程是不连续的|
|Score Matching||[link](./diffusion-tutorial-part/Fundamentals/Score-basedGenerativeModelingwithDifferentialEquations.md)|
|DSDFM|std normal --(flow matching/score matching)--> VQ-VAE latent --(VQ-VAE)--> pixel|[link](https://caterpillarstudygroup.github.io/ReadPapers/92.html)|

共同点：都是基于连续时间马尔可夫过程的随机过程Continuous-time Markov process。      

![](./assets/P10图1-0.png)

> \\(\Phi\\) 是从一次生成到另一次生成的转移函数。    
**增量生成模型的目标是学习转移函数**。      

