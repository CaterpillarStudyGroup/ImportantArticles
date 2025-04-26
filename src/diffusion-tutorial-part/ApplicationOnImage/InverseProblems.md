P3   
# 任务描述

1. 图像去噪
2. 图像超分
3. 图像补全

输入：  
![](../../assets/D3-3.png)  ![](../../assets/D3-4-2.png)  
输出：  
![](../../assets/D3-4.png)  
![](../../assets/D3-3-1.png) 

基于某个预训练的diffusion model，在无condition的情况下，每张图像都符合diffusion生成模型的分布。当以某个特定的图像（模糊图像、低分辨率图像）时，期望能够得到的是对应的清晰、高分辨率的图像的分布。  

P6   
# Replacement-based Methods

(Overwrites model prediction with known information)    

![](../../assets/D3-6.png)  

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2021|ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models||||
|||Kawar et al., <u>"SNIPS: Solving Noisy Inverse Problems Stochastically",</u> NeurIPS 2021   
|||Chung et al., <u>"Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction",</u> CVPR 2022    
|||Song et al., <u>"Solving Inverse Problems in Medical Imaging with Score-Based Generative Models",</u> ICLR 2022   
|||Kawar et al., <u>"Denoising Diffusion Restoration Models",</u> NeurIPS 2022   

P7   
# Reconstruction-based Methods

(Approximate classifier-free guidance **without additional training**)    
![](../../assets/D3-5-2.png)  

![](../../assets/D3-7.png)  

> Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023   

> &#x2705; cfg 使用\\((x,t)\\)的 pair data 来近似 \\(\nabla _{x_t} \log p_t(\mathbf{y}|\mathbf{x}_t)\\)，但此处没有 pair data，希望通过非训练的方法来得出。    
> &#x2705; 公式基于马尔可夫推导。\\(p(\mathbf{y}|\mathbf{x}_t)\\) 可描述为 \\(p(\mathbf{y}|\mathbf{x}_0)\\) 的期望。然后把期望从外面移到里面。    

P8    
In the Gaussian case,    

$$
p(\mathbf{y} |\mathbb{E} [\mathbf{x} _ 0|\mathbf{x} _ t])=-c||\mathcal{A} \mathbf{(\hat{x}}  _ 0)-\mathbf{y} ||^2_2
$$

**Maximizing the likelihood is minimizing the L2 distance between measured and generated!**     

![](../../assets/D3-8.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023    

> &#x2705; 在 diffusion 的同时做重建。    

## More

 - **Video Diffusion/Pyramid DDPM**: used for uper-resolution.      
 - **Pseudoinverse guidance**: linear and some non-differentiable problems, e.g., JPEG
 - **MCG**: combines replacement & reconstruction for linear problems.

**Others**
 - **CSGM**: Posterior sampling with Langevin Dynamics based on the diffusion score model.   
 - **RED-Diff**: A Regularizing-by-Denoising (RED), variational inference approach.   
 - **Posterior sampling**: use RealNVP to approximate posterior samples from diffusion models.   


|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|||Chung et al., <u>"Improving Diffusion Models for Inverse Problems using Manifold Constraints",</u> NeurIPS 2022   
|||Ryu and Ye, <u>"Pyramidal Denoising Diffusion Probabilistic Models",</u> arXiv 2022   
|||Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> arXiv 2022   
|||Song et al., <u>"Pseudoinverse-Guided Diffusion Models for Inverse Problems",</u> ICLR 2023   
|||Jalal et al., <u>"Robust Compressed Sensing MRI with Deep Generative Priors",</u> NeurIPS 2021   
|||Mardani et al., <u>"A Variational Perspective on Solving Inverse Problems with Diffusion Models",</u> arXiv 2023   
|||Feng et al., <u>"Score-Based Diffusion Models as Principled Priors for Inverse Imaging",</u> arXiv 2023   

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/