P2   
# Outline    
 - Inverse problems
 - Setup
 - Replacement-based methods
 - Reconstruction-based methods

P3   
# Diffusion Models for Inverse Problems

## Setup

### Goal: denoise and super-resolve an image

![](../../assets/D3-3.png)  

![](../../assets/D3-3-1.png)  

> &#x2705; 处理对象是一个特定的 Image． 虽然 Diffusion Prior 中包含大量高质量图像，但对于这个特定的处理对象，应当基于其中的部分 prior.    

P4   
### Goal: recover the masked region of an image

![](../../assets/D3-4-2.png)  

![](../../assets/D3-4.png)  

We want to use the same diffusion model for different problems!    

> &#x2705; super-resolve 和 inpaint．    

P5   
## Diffusion Models for Inverse Problems: Two Paradigms

![](../../assets/D3-5-1.png)  

**Replacement-based methods**    
(Overwrites model prediction with known information)    

![](../../assets/D3-5-2.png)  

**Reconstruction-based methods**    
(Approximate classifier-free guidance **without additional training**)    

> &#x2705; 第 3 种方法，在 noise 上覆盖，可以得到类似 reconstruction-based 方法的效果。    
> &#x2753; 重建方法与 cfg 有什么关系？答：见P7．     

P6   
## Replacement-based Methods: An Example   

![](../../assets/D3-6.png)  

Song et al., <u>"Score-Based Generative Modeling through Stochastic Differential Equations",</u> ICLR 2021    

P7   
## Reconstruction-based Methods: An Example

![](../../assets/D3-7.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023   

> &#x2705; cfg 使用\\((x,t)\\)的 pair data 来近似 \\(\nabla _{x_t} \log p_t(\mathbf{y}|\mathbf{x}_t)\\)，但此处没有 pair data，希望通过非训练的方法来得出。    
> &#x2705; 公式基于马尔可夫推导。\\(p(\mathbf{y}|\mathbf{x}_t)\\) 可描述为 \\(p(\mathbf{y}|\mathbf{x}_0)\\) 的期望。然后把期望从外面移到里面。    

P8    
## Diffusion Posterior Sampling   

In the Gaussian case,    

$$
p(\mathbf{y} |\mathbb{E} [\mathbf{x} _ 0|\mathbf{x} _ t])=-c||\mathcal{A} \mathbf{(\hat{x}}  _ 0)-\mathbf{y} ||^2_2
$$

Maximizing the likelihood is minimizing the L2 distance between measured and generated!     

![](../../assets/D3-8.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023    

> &#x2705; 在 diffusion 的同时做重建。    


P9   
## Solving Inverse Problems with Diffusion Models   

**Reconstruction-based methods**    
 - **ScoreSDE**: simple linear problems, e.g., inpainting, colorization; later extended to MRI and CT.   
 - **ILVR**: more linear problems, e.g., super-resolution.   
 - **SNIP**S: slow solution for noisy linear problems.   
 - **CCDF**: better initializations.    
 - **DDRM**: fast solution for all noisy linear problems, and JPEG.   

Choi et al., <u>"ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models",</u> ICCV 2021       
Kawar et al., <u>"SNIPS: Solving Noisy Inverse Problems Stochastically",</u> NeurIPS 2021   
Chung et al., <u>"Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction",</u> CVPR 2022    
Song et al., <u>"Solving Inverse Problems in Medical Imaging with Score-Based Generative Models",</u> ICLR 2022   
Kawar et al., <u>"Denoising Diffusion Restoration Models",</u> NeurIPS 2022   

P10    
## Solving Inverse Problems with Diffusion Models

**Replacement-based methods**    
 - **Video Diffusion/Pyramid DDPM**: used for uper-resolution.      
 - **Pseudoinverse guidance**: linear and some non-differentiable problems, e.g., JPEG
 - **MCG**: combines replacement & reconstruction for linear problems.

**Others**
 - **CSGM**: Posterior sampling with Langevin Dynamics based on the diffusion score model.   
 - **RED-Diff**: A Regularizing-by-Denoising (RED), variational inference approach.   
 - **Posterior sampling**: use RealNVP to approximate posterior samples from diffusion models.   

Ho et al., <u>"Video Diffusion Models",</u> NeurIPS 2022   
Chung et al., <u>"Improving Diffusion Models for Inverse Problems using Manifold Constraints",</u> NeurIPS 2022   
Ryu and Ye, <u>"Pyramidal Denoising Diffusion Probabilistic Models",</u> arXiv 2022   
Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> arXiv 2022   
Song et al., <u>"Pseudoinverse-Guided Diffusion Models for Inverse Problems",</u> ICLR 2023   
Jalal et al., <u>"Robust Compressed Sensing MRI with Deep Generative Priors",</u> NeurIPS 2021   
Mardani et al., <u>"A Variational Perspective on Solving Inverse Problems with Diffusion Models",</u> arXiv 2023   
Feng et al., <u>"Score-Based Diffusion Models as Principled Priors for Inverse Imaging",</u> arXiv 2023   





---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/