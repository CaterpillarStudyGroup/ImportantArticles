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

![](../assets/D3-3.png)  

![](../assets/D3-3-1.png)  

> &#x2705; 处理对象是一个特定的 Image． 虽然 Diffusion Prior 中包含大量高质量图像，但对于这个特定的处理对象，应当基于其中的部分 prior.    

P4   
### Goal: recover the masked region of an image

![](../assets/D3-4-2.png)  

![](../assets/D3-4.png)  

We want to use the same diffusion model for different problems!    

> &#x2705; super-resolve 和 inpaint．    

P5   
## Diffusion Models for Inverse Problems: Two Paradigms

![](../assets/D3-5-1.png)  

**Replacement-based methods**    
(Overwrites model prediction with known information)    

![](../assets/D3-5-2.png)  

**Reconstruction-based methods**    
(Approximate classifier-free guidance **without additional training**)    

> &#x2705; 第 3 种方法，在 noise 上覆盖，可以得到类似 reconstruction-based 方法的效果。    
> &#x2753; 重建方法与 cfg 有什么关系？答：见P7．     

P6   
## Replacement-based Methods: An Example   

![](../assets/D3-6.png)  

Song et al., <u>"Score-Based Generative Modeling through Stochastic Differential Equations",</u> ICLR 2021    

P7   
## Reconstruction-based Methods: An Example

![](../assets/D3-7.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023   

> &#x2705; cfg 使用\\((\mathbf{x,t})\\)的 pair data 来近似 \\(\nabla _{xt} \log p_t(\mathbf{y}|\mathbf{x}_t)\\)，但此处没有 pair data，希望通过非训练的方法来得出。    
> &#x2705; 公式基于马尔可夫推导。\\(p(\mathbf{y}|\mathbf{x}_t)\\) 可描述为 \\(p(\mathbf{y}|\mathbf{x}_0)\\) 的期望。然后把期望从外面移到里面。    

P8    
## Diffusion Posterior Sampling   

In the Gaussian case,    

$$
p(\mathbf{y} |\mathbb{E} [\mathbf{x} _ 0|\mathbf{x} _ t])=-c||\mathcal{A} \mathbf{(\hat{x}}  _ 0)-\mathbf{y} ||^2_2
$$

Maximizing the likelihood is minimizing the L2 distance between measured and generated!     

![](../assets/D3-8.png)  

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

P11   
# 3D    
 - **Diffusion on various 3D representations**    
 - 2D diffusion models for 3D generation    
 - Diffusion models for view synthesis    
 - 3D reconstruction    
 - 3D editing    

P12   
## Diffusion Models for Point Clouds   

A set of points with location information.    

![](../assets/D3-12.png)  

![](../assets/D3-12-1.png)  

Zhou et al., <u>"3D Shape Generation and Completion through Point-Voxel Diffusion",</u> ICCV 2021    
Liu et al, <u>"Point-Voxel CNN for Efficient 3D Deep Learning",</u> NeurIPS 2019    

> &#x2705; 分支1：逐顶点的 MLP (对应图中 b)   
> &#x2705; 分支2：VOX 可以看作是低分辨率的 points    
> &#x2705; 优点是结构化，可用于 CNN    
> &#x2753; VOX → points，低分辨到高分辨率要怎么做？   
> &#x2753; 怎么用把 voxel 内的点转换为 voxel 的特征？     

P13    
## Diffusion Models for Point Clouds    

![](../assets/D3-13.png)  

Zhou et al., <u>"3D Shape Generation and Completion through Point-Voxel Diffusion",</u> ICCV 2021    

> &#x2705; Completion：深度图 → 完整点    
> &#x2705; 方法：(1) 基于深度图生成点云 (2) 用 inpainting 技术补全    
> &#x2705; generation 和 completion 是两种不同的 task.    

P14     
## Diffusion Models for Point Clouds   

![](../assets/D3-14.png)  

Zeng et al., <u>"LION: Latent Point Diffusion Models for 3D Shape Generation",</u> NeurIPS 2022    

> &#x2705; 特点：    
> &#x2705; 1、latent diffusion model for point clouds.    
> &#x2705; 2、point-voxel CNN 架构，用于把 shape 编码成 latent shape 及 lantent point.    
> &#x2705; 3、diffusion model 把 latent point 重建出原始点。    

P15   
## Diffusion Models for Point Clouds

Point-E uses a synthetic view from fine-tuned GLIDE, and then ”lifts” the image to a 3d point cloud.

![](../assets/D3-15-1.png)  

![](../assets/D3-15-2.png)  

Nichol et al., <u>"Point-E: A System for Generating 3D Point Clouds from Complex Prompts",</u> arXiv 2022     

> &#x2705; point E task：文生成点云。    
> &#x2705; 第1步：文生图，用 fine-tuned GLIDE     
> &#x2705; 第2步：图生点，用 transformer-based diffusion model.     

P16   
## Diffusion Models for Signed Distance Functions   

SDF is a function representation of a surface.  
For each location x, |SDF(x)| = smallest distance to any point on the surface.    

![](../assets/D3-16.png)  

P17   
## Diffusion Models for Signed Distance Functions   

 - Memory of SDF grows cubically with resolution    
 - Wavelets can be used for compression!   
 - Diffusion for coarse coefficients, then predict detailed ones.   

![](../assets/D3-17-1.png)  

![](../assets/D3-17-2.png)  

Hui et al., <u>"Neural Wavelet-domain Diffusion for 3D Shape Generation",</u> arXiv 2022    

> &#x2705; 这里说的 SDF，是用离散的方式来记录每个点的 distance.     
> &#x2705; Wavelet 把 SDF 变为 coarse 系数，diffusion model 生成 coarse系数，再通过另一模型变为 detailed   

P18   
## Diffusion Models for Signed Distance Functions

![](../assets/D3-18.png)  

**Latent space diffusion for SDFs, where conditioning can be provided with cross attention**

Chou et al., <u>"DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions",</u> arXiv 2022    

> &#x2705; 原理与上一页相似，只是把 waveles 换成了 VAE.    

P19   
## Diffusion Models for Other 3D Representations    

Neural Radiance Fields (NeRF) is another representation of a 3D object.    

![](../assets/D3-19.png)  

> &#x2705; NeRF：用体的方式来描述 3D 物体     
> &#x2705; (1) 从 diffusion 中提取 image （2）从 image 计算 loss (3) loss 更新 image (4) image 更新 NeRF．    
> &#x2705; \\(（x,y,z,\theta ,\phi ）\\) 是每个点在向量中的表
示，其中前三维是 world coordinate，后面两维是 viewing direction     
> &#x2705; density 描述这个点有多透明。    
> &#x2705; F 是一个小型的网络，例如 MLP.    

 
P20   
## Diffusion Models for Other 3D Representations

![](../assets/D3-20-1.png)  
**NeRF**    
(Fully implicit)    

![](../assets/D3-20-2.png)  
**Voxels**    
(Explicit / hybrid)    

![](../assets/D3-20-3.png)  
**Triplanes**    
(Factorized, hybrid)    

Image from EG3D paper.    

> &#x2705; Nenf 可以有三种表示形式    

P21   
## Diffusion Models for Other 3D Representations     

 - Triplanes, regularized ReLU Fields, the MLP of NeRFs...    
 - A good representation is important!     

![](../assets/D3-21-1.png)  
Triplane diffusion    

![](../assets/D3-21-2.png)  
Regularized ReLU Fields    

![](../assets/D3-21-3.png)  
Implicit MLP of NeRFs     


Shue et al., <u>"3D Neural Field Generation using Triplane Diffusion",</u> arXiv 2022    
Yang et al., <u>"Learning a Diffusion Prior for NeRFs",</u> ICLR Workshop 2023    
Jun and Nichol, <u>"Shap-E: Generating Conditional 3D Implicit Functions",</u> arXiv 2023    

> &#x2705; 这三种表示形式都可以与 diffuson 结合好的表示形式对diffusion 的效果很重要。    

P22    
## Outline


 - 2D diffusion models for 3D generation     

P23   
## 2D Diffusion Models for 3D Generation   

 - Just now, we discussed diffusion models directly on 3d.   
 - However, there are a lot fewer 3d data than 2d.   
    - A lot of experiments are based on ShapeNet!   
 - Can we use 2d diffusion models as a “prior” for 3d?   

P25    
## DreamFusion: Setup    

 - Suppose there is a text-to-image diffusion model.    
 -  Goal: optimize NeRF parameter such that each angle “looks 
good” from the text-to-image model.    
 - Unlike ancestral sampling (e.g., DDIM), the underlying 
parameters are being optimized over some loss function.    

![](../assets/D3-25-1.png)     
![](../assets/D3-25-2.png)     

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023    

> &#x2705; 参数不在 2D 空间而是在 Nerf 空间，构成优化问题，通过更新 Nerf 参数来满足 loss.    
![](../assets/D3-25-3.png)  



P26   
## DreamFusion: Score Distillation Sampling   

![](../assets/D3-26.png)     

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023     


> &#x2705; 第二项：\\( \partial \\) Output／ \\( \partial \\) Input．   
> &#x2705; 第三项：\\( \partial \\) Input Image／ \\( \partial \\)  Nerf Angle    
> &#x2705; 第二项要计算 diffusion model 的梯度，成本非常高。    
> &#x2705; 第二项要求反向传播 through the diffuson model，很慢，且费内存。   
> &#x2705; 这一页描述 Image → Loss 的过程。    
> &#x2705; 公式 1 为 diffusion model objective fanction.    
> &#x2705; 公式 2 为算法中使用的 loss，由于\\(\lambda =g(\theta )\\)，\\(\frac{\partial L}{\partial \theta } =\frac{\partial L}{\partial \lambda } \cdot \frac{\partial \lambda }{\partial \theta } \\)，其中 \\(\frac{\partial L}{\partial \lambda }\\) 又分为第一项和第二项。    
> &#x2705; 公式 2 中的常系数都省掉了。    
> &#x2705; 公式 3 把公式 2 中的第二项去掉了，为本文最终使用的 loss.   

P27   
## DreamFusion: Score Distillation Sampling

Consider the KL term to minimize (given t):   

$$
\mathbf{KL} (q(\mathbf{z} _ t|g(\theta );y,t)||p\phi (\mathbf{z} _ t;y,t))
$$

KL between noisy real image distribution and generated image 
distributions, conditioned on y!     

KL and its gradient is defined as:    

![](../assets/D3-27.png)  

(B) can be derived from chain rule    

$$
\nabla _ \theta \log p _ \phi (\mathbf{z} _ t|y)=s _ \phi (\mathbf{z} _ t|y)\frac{\partial \mathbf{z} _ t}{\partial \theta }=\alpha _ ts _ \phi (\mathbf{z} _ t|y)\frac{\partial \mathbf{x} }{\partial \theta } =-\frac{\alpha _ t}{\sigma _ t}\hat{\epsilon }_ \phi (\mathbf{z} _ t|y)\frac{\partial \mathbf{x} }{\partial \theta }   
$$

(A) is the gradient of the entropy of the forward process with fixed variance = 0.    

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023   

> &#x2705; A: the gradient of the entropy of the forward process。由于前向只是加噪，因此 A 是固定值。即 0.    
> &#x2705; P27 和 P28 证明 P26 中的第二项可以不需要。  
> &#x2753; KL 散度用来做什么？LOSS 里没有这一项。    
> &#x2705; KL 用于度量 \\(P(Z_t｜t)\\) 和 \\(q(Z_t｜t)\\)．
> &#x2705; KL 第一项为 Nerf 的渲染结果加噪，KL 第二项为真实数据加噪。    

P28    
## DreamFusion: Score Distillation Sampling  

$$
(A)+(B)=\frac{\alpha _ t}{\sigma _ t}\hat{\epsilon } _ \phi (\mathbf{z} _ t|y)\frac{\partial \mathbf{x} }{\partial \theta }
$$

However, this objective can be quite noisy.     
Alternatively, we can consider a “baseline” approach in reinforcement learning: add a component that has zero mean but reduces variance. Writing out (A) again:     

![](../assets/D3-28-1.png)  

Thus, we have:

![](../assets/D3-28-2.png)  

This has the same mean, but **reduced variance**, as we train \\(\hat{\epsilon } _ \phi\\) to predict \\(\epsilon\\)    


Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023    

P29   
## DreamFusion in Text-to-3D    

 - SDS can be used to optimize a 3D representation, like NeRF.   

![](../assets/D3-29.png)  

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023    

> &#x2705; (1) 生成 Nerf (2) Nerf 投影 (3) 投影图加噪再去噪 (4) 对生成结果求 loss    
> &#x2705; entire pipeline.    
> &#x2705; 左中上：从相机视角，生成 object 的投影。   
> &#x2705; 左下：以相机视角为参数，推断出每个点的 Nerf 参数。   
> &#x2705; 左中：左上和左下结合，得到渲染图像。    
> &#x2705; 生成随机噪声，对渲染图像加噪。   
> &#x2705; 右上：使用 diffusion model 从加噪图像中恢复出原始图像。（包含多个 step）   
> &#x2705; 右下：得到噪声，并与原始噪声求 loss.    
> &#x2705; 根据 loss 反传梯度优化左下的 MLP.    

P30   
## Extensions to SDS: Magic3D

2x speed and higher resolution   
 - Accelerate NeRF with Instant-NGP, for coarse representations.    
 - Optimize a fine mesh model with differentiable renderer.   

![](../assets/D3-30.png)  

Lin et al., <u>"Magic3D: High-Resolution Text-to-3D Content Creation",</u> CVPR 2023   

> &#x2705; Instant NGP 代替左下的 Nerf MLP．以 coarse representetion 作为 condition 来生成 fine mesh model.    

P31
## Alternative to SDS: Score Jacobian Chaining

A different formulation, motivated from approximating 3D score.   

![](../assets/D3-31.png)  

In principle, the diffusion model is the noisy 2D score (over clean images),   
but in practice, the diffusion model suffers from out-of-distribution (OOD) issues!    

For diffusion model on noisy images, **the non-noisy images are OOD**!    

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.     

> &#x2705; 2D sample, 3D score    

P32   
## Score Jacobian Chaining   

SJC approximates noisy score with “Perturb-and-Average Scoring”, which is not present in SDS.   
 - Use score model on multiple noise-perturbed data, then average it.    

![](../assets/D3-32.png)  

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.    

> &#x2705; 通过这种方法来近似 clean image 的输出，解决 clean image 的 OOD 问题。    


P33    
## SJC and SDS

SJC is a competitive alternative to SDS.   

![](../assets/D3-33.png) 

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.    

P34   
## Alternative to SDS: ProlificDreamer   

 - SDS-based method often set classifier-guidance weight to 100, which limits the “diversity” of the generated samples.   
 - ProlificDreamer reduces this to 7.5, leading to diverse samples.    

![](../assets/D3-34.png) 

Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

P35   
## ProlificDreamer and Variational Score Distillation  

Instead of maximizing the likelihood under diffusion model, VSD minimizes the KL divergence via variational inference.    

$$
\begin{matrix}
\min_{\mu } D _ {\mathrm{KL} }(q^\mu _ 0(\mathbf{x} _ 0|y)||p _ 0(\mathbf{x} _ 0|y)). \\\\
\quad \mu \quad \text{is the distribution of NeRFs} .
\end{matrix}
$$

Suppose is a \\(\theta _ \tau \sim \mu \\) NeRF sample, then VSD simulates this ODE:    

![](../assets/D3-35.png) 

 - Diffusion model can be used to approximate score of noisy real images.   
 - How about noisy rendered images?   sss

> &#x2705; 第一项由 diffusion model 得到，在此处当作 GT．   

P36   
## ProlificDreamer and Variational Score Distillation

 - Learn another diffusion model to approximate the score of noisy rendered images!

![](../assets/D3-36.png) 

Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

> &#x2705; 使用 LoRA 近第二项。    

P37   
## Why does VSD work in practice?    

 - The valid text-to-image NeRFs form a distribution with infinite possibilities!    
 - In SDS, epsilon is the score of noisy “dirac distribution” over finite renders, which converges to the true score with infinite renders!    
 - In VSD, the LoRA model aims to **represent the (true) score of noisy distribution over infinite number of renders!**   
 - If the generated NeRF distribution is only one point and LoRA overfits perfectly, then VSD = SDS!    
 - But LoRA has good generalization (and learns from a trajectory of NeRFs), so closer to the true score!    

 - This is analogous to    
    - Representing the dataset score via mixture of Gaussians on the dataset (SDS), versus     
    - Representing the dataset score via the LoRA UNet (VSD)    


Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

P38   
## Outline

 - Diffusion models for view synthesis   

P39
## Novel-view Synthesis with Diffusion Models   

 - These do not produce 3D as output, but synthesis the view at different angles.    

Watson et al., <u>"Novel View Synthesis with Diffusion Models",</u> ICLR 2023    

P40   
## 3DiM   

 - Condition on a frame and two poses, predict another frame.     

![](../assets/D3-40-1.png)  

UNet with frame cross-attention   

![](../assets/D3-40-2.png)  

Sample based on stochastic conditions,   
allowing the use of multiple conditional frames.    


Watson et al., <u>"Novel View Synthesis with Diffusion Models",</u> ICLR 2023    

> &#x2705; UNet，2 branch，分别用于原始角度和要生成的角度。   
> &#x2705; 引入 step 2 是为了内容一致性。   
> &#x2705; frame：坐标系。在不同的坐标系下看到的是不同的视角。    
> &#x2753; 为什么有两个pose？
> &#x2705; 每个 frame 的内部由 cross-attention 连接。    

P41    
## GenVS   

 - 3D-aware architecture with latent feature field.    
 - Use diffusion model to improve render quality based on structure.   

![](../assets/D3-41.png)  

Chan et al., <u>"Generative Novel View Synthesis with 3D-Aware Diffusion Models",</u> arXiv 2023    

> &#x2705; (1) 生成 feature field (2) render 其中一个视角 (3) 优化渲染效果     
> &#x2705; (2) 是 MLP (3) 是 diffusion．    

P42    
## Outline   

 - 3D reconstruction    

P43    
## NeuralLift-360 for 3D reconstruction

 - SDS + Fine-tuned CLIP text embedding + Depth supervision    

![](../assets/D3-43.png)  

Xu et al., <u>"NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views",</u> CVPR 2023    

> &#x2705; 整体上是类似 SDS 的优化方法，再结合其它的损失函数。    
> &#x2705; (1) 渲染不同视角，并对渲染结果用 clip score打分。    
> &#x2705; (2) 监督深度信息。    

P44    
## Zero 1-to-3   

 - Generate novel view from 1 view and pose, with 2d model.    
 - Then, run SJC / SDS-like optimizations with view-conditioned model.   

![](../assets/D3-44.png)  

Liu et al., <u>"Zero-1-to-3: Zero-shot One Image to 3D Object",</u> arXiv 2023    

> &#x2705; (1) 用 2D diffusion 生成多视角。用 SDS 对多视角图像生成3D．   

P45    
## Outline

 - Inverse problems    
 - **3D**    
 - Video    
 - Miscellaneous    

 - 3D editing    

P46   
## Instruct NeRF2NeRF

Edit a 3D scene with text instructions   

![](../assets/D3-46.png)  

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023      

> &#x2705; 用 Nerf 来描述 3D scene。通过文件条件把原 Nerf，变成另一个 Nerf，从而得到新的 3D scene.    

P47   
## Instruct NeRF2NeRF   

**Edit a 3D scene with text instructions**   
 -  Given existing scene, use Instruct Pix2Pix to edit image at different viewpoints.   
 - Continue to train the NeRF and repeat the above process   

![](../assets/D3-47.png)  

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023   

> &#x2705; 首先有一个训好的 Nerf. 对一个特定的场景使用 Instruct Pix 2 pix 在 2D 上编辑训练新的 Werf.    
> &#x2705; 基于 score disllation sampling.    

P48   
## Instruct NeRF2NeRF

With each iteration, the edits become more consistent.    

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023   

P49   
## Vox-E: Text-guided Voxel Editing of 3D Objects    

 - Text-guided object editing with SDS
 - Regularize the structure of the new voxel grid.

![](../assets/D3-49.png)  

Sella et al., <u>"Vox-E: Text-guided Voxel Editing of 3D Objects",</u> arXiv 2023   

P50   
# Video   

- Video generative models   

P51   
## Video Diffusion Models

3D UNet from a 2D UNet.   
 - 3x3 2d conv to 1x3x3 3d conv.   
 - Factorized spatial and temporal attentions.   

![](../assets/D3-51.png)     
![](../assets/D3-51-2.png)     

Ho et al., <u>"Video Diffusion Models",</u> NeurIPS 2022   

> &#x2705; 利用 2D 做 3D，因为 3D 难训且数据少。  
> &#x2705; 通到添加时序信息，把图像变成视频。此处的 3D 是 2D＋时间，而不是立体。对向量增加一个维度，并在新的维度上做 attention.   


P52   
## Imagen Video: Large Scale Text-to-Video   


 - 7 cascade models in total.   
 - 1 Base model (16x40x24)   
 - 3 Temporal super-resolution models.   
 - 3 Spatial super-resolution models.   

![](../assets/D3-52.png)     

Ho et al., <u>"Imagen Video: High Definition Video Generation with Diffusion Models",</u> 2022    


> &#x2705; 通过 7 次 cascade，逐步提升顺率和像素的分辨率，每一步的训练对上一步是依赖的。    

P53   
## Make-a-Video  

 - Start with an unCLIP (DALL-E 2) base network.    

![](../assets/D3-53.png)     

Singer et al., <u>"Make-A-Video: Text-to-Video Generation without Text-Video Data",</u> ICLR 2023    

> &#x2705; 把 text 2 image model 变成 text to video model，但不需要 text-video 的 pair data.    

P54   
## Make-a-Video   

![](../assets/D3-54-1.png)     
3D Conv from Spatial Conv + Temporal Conv   

![](../assets/D3-54-2.png)     
3D Attn from Spatial Attn + Temporal Attn   

Different from Imagen Video, only the image prior takes text as input!   


Singer et al., <u>"Make-A-Video: Text-to-Video Generation without Text-Video Data",</u> ICLR 2023     

> &#x2705; 与 Imagin 的相同点：(1) 使用 cascade 提升分辨率， (2) 分为时间 attention 和空间 attention.    
> &#x2705; 不同点：(1) 时间 conv＋空间 conv.    

P55   
## Video LDM   

 - Similarly, fine-tune a text-to-video model from text-to-image model.    

![](../assets/D3-55.png)     

Blattmann et al., <u>"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models",</u> CVPR 2023    

> &#x2705; 特点：(1) 使用 latent space． (2) Encoder fix．用 video 数据 fineture Decoder.    

P56   
## Video LDM: Decoder Fine-tuning   

 - Fine-tune the decoder to be video-aware, keeping encoder frozen.    

![](../assets/D3-56.png)     

Blattmann et al., <u>"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models",</u> CVPR 2023    

P57    
## Video LDM: LDM Fine-tuning   

 - Interleave spatial layers and temporal layers.    
 - The spatial layers are frozen, whereas temporal layers are trained.    
 - Temporal layers can be Conv3D or Temporal attentions.   
 - Context can be added for autoregressive generation.    

![](../assets/D3-57.png)     

Optional context via learned down-sampling operation.   

**For Conv3D,**    
shape is [batch, time, channel, height, width]    

**For Temporal attention,**   
shape is [batch \\(^\ast \\)height\\(^\ast \\) width, time, channel]    



Blattmann et al., <u>"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models",</u> CVPR 2023   

> &#x2705; 时序层除了时序 attention，还有 3D conv，是真正的 3D，但是更复杂，且计算、内存等消耗更大。   


P58   
## Video LDM: Upsampling
 - After key latent frames are generated, the latent frames go through temporal interpolation.   
 - Then, they are decoded to pixel space and optionally upsampled.   

![](../assets/D3-58.png)     

Blattmann et al., <u>"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models",</u> CVPR 2023    

P59   
## Outline   

 - Video style transfer / editing methods    

P60   
## Gen-1

 - Transfer the style of a video using text prompts given a “driving video”

![](../assets/D3-60.png)     

Esser et al., <u>"Structure and Content-Guided Video Synthesis with Diffusion Models",</u> arXiv 2023    

P61   
## Gen-1

 - Condition on structure (depth) and content (CLIP) information.   
 - Depth maps are passed with latents as input conditions.   
 - CLIP image embeddings are provided via cross-attention blocks.   
 - During inference, CLIP text embeddings are converted to CLIP image embeddings.    

![](../assets/D3-61.png)     

Esser et al., <u>"Structure and Content-Guided Video Synthesis with Diffusion Models",</u> arXiv 2023     

> &#x2705; 用 depth estimator 从源视频提取 struct 信息，用 CLIP 从文本中提取 content 信息。   
> &#x2705; depth 和 content 分别用两种形式注入。depth 作为条件，与 lantent concat 到一起。content 以 cross attention 的形式注入。    


P62   
## Pix2Video: Video Editing Using Image Diffusion   

 - Given a sequence of frames, generate a new set of images that reflects an edit.   
 - Editing methods on individual images fail to preserve temporal information.    

Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023    

> &#x2705; 没有 3D diffusion model，只是用 2D diffusion model 生成多张图像并拼成序列。关键在于保持时序的连续性。    

P63   
## Pix2Video: Video Editing Using Image Diffusion   

![](../assets/D3-63-1.png)    

 - **Self-Attention injection:** use the features of previous frame for Key and Values.   

![](../assets/D3-63-2.png)    

Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023   

> &#x2705; (1) 使用 DDIM inversion 把图像转为 noise．   
> &#x2705; (2) 相邻的 fram 应 inversion 出相似的 noise．    
> &#x2705; 使用 self-attention injection 得到相似的 noise.    


P64    
## Pix2Video: Video Editing Using Image Diffusion   

![](../assets/D3-64.png)    

Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023    

> &#x2705; reconstruction guidance，使生成的 latent code 与上一帧接近。    

P65   
## Pix2Video: Video Editing Using Image Diffusion  

 - The two methods improve the temporal consistency of the final video!   

Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023   

P66    
## Concurrent / Related Works   

![](../assets/D3-66-1.png)    
Video-P2P: Cross-Attention Control on text-to-video model    


![](../assets/D3-66-2.png)    
FateZero: Store attention maps from DDIM inversion for later use    


![](../assets/D3-66-3.png)    
Tune-A-Video: Fine-tune projection matrices of the attention layers, from text2image model to text2video model.    


![](../assets/D3-66-4.png)    
Vid2vid-zero: Learn a null-text embedding for inversion, then use cross-frame attention with original weights.    


P67   
# Miscellaneous   

 - Diffusion models for large contents   


P68   
## Diffusion Models for Large Contents   

 - Suppose model is trained on small, squared images, how to extend it to larger images?   
 - Outpainting is always a solution, but not a very efficient one!   

Let us generate this image with a diffusion model only trained on squared regions:    

![](../assets/D3-68-1.png)    

1. Generate the center region \\(q(\mathbf{x} _ 1,\mathbf{x} _ 2)\\)    
2. Generate the **surrounding region conditioned on parts of the center image** \\(q(\mathbf{x} _ 3|\mathbf{x} _ 2)\\)    

![](../assets/D3-68-2.png)    

Latency scales linearly with the content size!     

> &#x2705; 根据左边的图生成右边的图，存在的问题：慢     
> &#x2705; 直接生成大图没有这样的数据。   
> &#x2705; 并行化的生成。    
   
P69   
## Diffusion Models for Large Contents

 - Unlike autoregressive models, diffusion models can generate large contents **in parallel**!    

![](../assets/D3-69-1.png)    



Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models",</u> CVPR 2023    

P70   
## Diffusion Models for Large Contents   

 - A “large” diffusion model from “small” diffusion models!   

![](../assets/D3-70-1.png)    

![](../assets/D3-70-2.png)    

Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models",</u> CVPR 2023   

P71   
## Diffusion Models for Large Contents

 - Applications such as long images, looped motion, 360 images…   

Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models",</u> CVPR 2023   

P72   
## Related Works   

 - Based on similar ideas but differ in how overlapping regions are mixed.

![](../assets/D3-72-1.png)    

![](../assets/D3-72-2.png)    

Jiménez, <u>"Mixture of Diffusers for scene composition and high resolution image generation",</u> arXiv 2023    
Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> ICML 2023   

> &#x2705; 这种并行化方法可以用于各种 overlapping 的场景。    

P73   
## Outline

 - Safety and limitations of diffusion models   

P74   
## Data Memorization in Diffusion Models

 - Due to the likelihood-base objective function, **diffusion models can ”memorize” data**.    
 - And with a higher chance than GANs!   
 - Nevertheless, a lot of “memorized images” are highly-duplicated in the dataset.    

![](../assets/D3-74.png)  

Carlini et al., <u>"Extracting Training Data from Diffusion Models",</u> arXiv 2023    

P75   
## Erasing Concepts in Diffusion Models   

 - Fine-tune a model to remove unwanted concepts.   
 - From original model, **obtain score via negative CFG**.   
 - **A new model is fine-tuned** from the new score function.    

![](../assets/D3-75-1.png)  

![](../assets/D3-75-2.png)  

Gandikota et al., <u>"Erasing Concepts from Diffusion Models",</u> arXiv 2023    

> &#x2705; 考虑到版权等问题。    
> &#x2705; finetune 已有的 text-2-image model．   
> &#x2705; 使用 negative CFG 原有信息不会受到影响。    


# Reference
P77   
## Part I

Ho et al., <u>"Denoising Diffusion Probabilistic Models",</u> NeurIPS 2020    
Song et al., <u>"Score-Based Generative Modeling through Stochastic Differential Equations",</u> ICLR 2021   
Kingma et al., <u>"Variational Diffusion Models",</u> arXiv 2021   
Karras et al., <u>"Elucidating the Design Space of Diffusion-Based Generative Models",</u> NeurIPS 2022   
Song et al., <u>"Denoising Diffusion Implicit Models",</u> ICLR 2021   
Jolicoeur-Martineau et al., "Gotta Go Fast When Generating Data with Score-Based Models",</u> arXiv 2021   
Liu et al., <u>"Pseudo Numerical Methods for Diffusion Models on Manifolds",</u> ICLR 2022   
Lu et al., <u>"DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps",</u> NeurIPS 2022   
Lu et al., <u>"DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models",</u> NeurIPS 2022   
Zhang and Chen, <u>"Fast Sampling of Diffusion Models with Exponential Integrator",</u> arXiv 2022   
Zhang et al., <u>"gDDIM: Generalized denoising diffusion implicit models",</u> arXiv 2022   
Zhao et al., <u>"UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models",</u> arXiv 2023    
Shih et al., <u>"Parallel Sampling of Diffusion Models",</u> arxiv 2023   
Chen et al., <u>"A Geometric Perspective on Diffusion Models",</u> arXiv 2023   
Xiao et al., <u>"Tackling the Generative Learning Trilemma with Denoising Diffusion GANs",</u> arXiv 2021   
Salimans and Ho, <u>"Progressive Distillation for Fast Sampling of Diffusion Models",</u> ICLR 2022   
Meng et al., <u>"On Distillation of Guided Diffusion Models",</u> arXiv 2022   
Dockhorn et al., <u>"GENIE: Higher-Order Denoising Diffusion Solvers",</u> NeurIPS 2022   
Watson et al., <u>"Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality",</u> ICLR 2022   
Phung et al., <u>"Wavelet Diffusion Models Are Fast and Scalable Image Generators",</u> CVPR 2023   
Dhariwal and Nichol, <u>"Diffusion Models Beat GANs on Image Synthesis",</u> arXiv 2021   
Ho and Salimans, <u>"Classifier-Free Diffusion Guidance",</u> NeurIPS Workshop 2021     
Automatic1111, <u>"Negative Prompt",</u> GitHub   
Hong et al., <u>"Improving Sample Quality of Diffusion Models Using Self-Attention Guidance",</u> arXiv 2022   
Saharia et al., <u>"Image Super-Resolution via Iterative Refinement",</u> arXiv 2021   
Ho et al., <u>"Cascaded Diffusion Models for High Fidelity Image Generation",</u> JMLR 2021   
Sinha et al., <u>"D2C: Diffusion-Denoising Models for Few-shot Conditional Generation",</u> NeurIPS 2021   
Vahdat et al., <u>"Score-based Generative Modeling in Latent Space",</u> arXiv 2021   
Rombach et al., <u>"High-Resolution Image Synthesis with Latent Diffusion Models",</u> CVPR 2022   
Daras et al., <u>"Score-Guided Intermediate Layer Optimization: Fast Langevin Mixing for Inverse Problems",</u> ICML 2022   

P78   
## Part I (cont’d)

Bortoli et al.,<u> "Diffusion Schrödinger Bridge",</u> NeurIPS 2021       
Bortoli et al.,<u> "Riemannian Score-Based Generative Modelling",</u> NeurIPS 2022  
Neklyudov et al., <u>"Action Matching: Learning Stochastic Dynamics from Samples",</u> ICML 2023  
Bansal et al., <u>"Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise",</u> arXiv 2022   
Daras et al., <u>"Soft Diffusion: Score Matching for General Corruptions",</u> TMLR 2023   
Delbracio and Milanfar, <u>"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration",</u> arXiv 2023   
Luo et al., <u>"Image Restoration with Mean-Reverting Stochastic Differential Equations",</u> ICML 2023   

P79    
## Part II

Bao et al., <u>"All are Worth Words: a ViT Backbone for Score-based Diffusion Models",</u> arXiv 2022    
Peebles and Xie, <u>"Scalable Diffusion Models with Transformers",</u> arXiv 2022    
Bao et al., <u>"One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale",</u> arXiv 2023    
Jabri et al., <u>"Scalable Adaptive Computation for Iterative Generation",</u> arXiv 2022    
Hoogeboom et al., <u>"simple diffusion: End-to-end diffusion for high resolution images",</u> arXiv 2023    
Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations",</u> ICLR 2022    
Li et al., <u>"Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models",</u> NeurIPS 2022   
Avrahami et al., <u>"Blended Diffusion for Text-driven Editing of Natural Images",</u> CVPR 2022     
Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control",</u> ICLR 2023    
Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models",</u> CVPR 2023    
Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance",</u> ICLR 2023    
Sarukkai et al., <u>"Collage Diffusion",</u> arXiv 2023    
Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> ICML 2023    
Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    
Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    
Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    
Tewel et al., <u>"Key-Locked Rank One Editing for Text-to-Image Personalization",</u> SIGGRAPH 2023    
Zhao et al., <u>"A Recipe for Watermarking Diffusion Models",</u> arXiv 2023    
Hu et al., <u>"LoRA: Low-Rank Adaptation of Large Language Models",</u> ICLR 2022    
Li et al., <u>"GLIGEN: Open-Set Grounded Text-to-Image Generation",</u> CVPR 2023    

P80    
## Part II (cont’d)

Avrahami et al., <u>"SpaText: Spatio-Textual Representation for Controllable Image Generation",</u> CVPR 2023    
Zhang and Agrawala, <u>"Adding Conditional Control to Text-to-Image Diffusion Models",</u> arXiv 2023    
Mou et al., <u>"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models",</u> arXiv 2023    
Orgad et al., <u>"Editing Implicit Assumptions in Text-to-Image Diffusion Models",</u> arXiv 2023    
Han et al., <u>"SVDiff: Compact Parameter Space for Diffusion Fine-Tuning",</u> arXiv 2023    
Xie et al., <u>"DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning",</u> arXiv 2023    
Saharia et al., <u>"Palette: Image-to-Image Diffusion Models",</u> SIGGRAPH 2022    
Whang et al., <u>"Deblurring via Stochastic Refinement",</u> CVPR 2022    
Xu et al., <u>"Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models",</u> arXiv 2023    
Saxena et al., <u>"Monocular Depth Estimation using Diffusion Models",</u> arXiv 2023    
Li et al., <u>"Your Diffusion Model is Secretly a Zero-Shot Classifier",</u> arXiv 2023    
Gowal et al., <u>"Improving Robustness using Generated Data",</u> NeurIPS 2021    
Wang et al., <u>"Better Diffusion Models Further Improve Adversarial Training",</u> ICML 2023    

P81   
## Part III   

Jalal et al., <u>"Robust Compressed Sensing MRI with Deep Generative Priors",</u> NeurIPS 2021  
Song et al., <u>"Solving Inverse Problems in Medical Imaging with Score-Based Generative Models",</u> ICLR 2022   
Kawar et al., <u>"Denoising Diffusion Restoration Models",</u> NeurIPS 2022   
Chung et al., <u>"Improving Diffusion Models for Inverse Problems using Manifold Constraints",</u> NeurIPS 2022   
Ryu and Ye, <u>"Pyramidal Denoising Diffusion Probabilistic Models",</u> arXiv 2022   
Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> arXiv 2022   
Feng et al., <u>"Score-Based Diffusion Models as Principled Priors for Inverse Imaging",</u> arXiv 2023   
Song et al., <u>"Pseudoinverse-Guided Diffusion Models for Inverse Problems",</u> ICLR 2023   
Mardani et al., <u>"A Variational Perspective on Solving Inverse Problems with Diffusion Models",</u> arXiv 2023   
Delbracio and Milanfar, <u>"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration",</u> arxiv 2023   
Stevens et al., <u>"Removing Structured Noise with Diffusion Models",</u> arxiv 2023   
Wang et al., <u>"Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model",</u> ICLR 2023   
Zhou et al., <u>"3D Shape Generation and Completion through Point-Voxel Diffusion",</u> ICCV 2021   
Zeng et al., <u>"LION: Latent Point Diffusion Models for 3D Shape Generation",</u> NeurIPS 2022   
Nichol et al., <u>"Point-E: A System for Generating 3D Point Clouds from Complex Prompts",</u> arXiv 2022   
Chou et al., <u>"DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions",</u> arXiv 2022   
Cheng et al., <u>"SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation",</u> arXiv 2022   
Hui et al., <u>"Neural Wavelet-domain Diffusion for 3D Shape Generation",</u> arXiv 2022   
Shue et al., <u>"3D Neural Field Generation using Triplane Diffusion",</u> arXiv 2022   
Yang et al., <u>"Learning a Diffusion Prior for NeRFs",</u> ICLR Workshop 2023   
Jun and Nichol, <u>"Shap-E: Generating Conditional 3D Implicit Functions",</u> arXiv 2023   
Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> arXiv 2022   
Lin et al., <u>"Magic3D: High-Resolution Text-to-3D Content Creation",</u> arXiv 2022   
Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> arXiv 2022   
Metzer et al., <u>"Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures",</u> arXiv 2022   
Hong et al., <u>"Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation",</u> CVPR Workshop 2023   
Watson et al., <u>"Novel View Synthesis with Diffusion Models",</u> arXiv 2022   
Chan et al., <u>"Generative Novel View Synthesis with 3D-Aware Diffusion Models",</u> arXiv 2023   
Xu et al., <u>"NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views",</u> arXiv 2022   
Zhou and Tulsiani, <u>"SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction",</u> arXiv 2022   

P82   
## Part III (cont’d)

Seo et al., <u>"DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model",</u> arXiv 2023   
Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023   
Sella et al., <u>"Vox-E: Text-guided Voxel Editing of 3D Objects",</u> arXiv 2023   
Ho et al., <u>"Video Diffusion Models",</u> NeurIPS 2022   
Harvey et al., <u>"Flexible Diffusion Modeling of Long Videos",</u> arXiv 2022    
Voleti et al., <u>"MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation",</u> NeurIPS 2022   
Ho et al., <u>"Imagen Video: High Definition Video Generation with Diffusion Models",</u> Oct 2022   
Singer et al., <u>"Make-A-Video: Text-to-Video Generation without Text-Video Data",</u> arXiv 2022    
Mei and Patel, <u>"VIDM: Video Implicit Diffusion Models",</u> arXiv 2022    
Blattmann et al., <u>"Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models",</u> CVPR 2023    
Wang et al., <u>"Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models",</u> arXiv 2023    
Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023    
Esser et al., <u>"Structure and Content-Guided Video Synthesis with Diffusion Models",</u> arXiv 2023    
Jiménez, <u>"Mixture of Diffusers for scene composition and high resolution image generation",</u> arXiv 2023    
Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> arXiv 2023    
Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models",</u> CVPR 2023    
Zhang et al., <u>"MotionDiffuse: Text-Driven Human Motion Generation with Diffusion Model",</u> arXiv 2022    
Tevet et al., <u>"Human Motion Diffusion Model",</u> arXiv 2022    
Chen et al., <u>"Executing your Commands via Motion Diffusion in Latent Space",</u> CVPR 2023    
Du et al., <u>"Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model",</u> CVPR 2023    
Shafir et al., <u>"Human Motion Diffusion as a Generative Prior",</u> arXiv 2023    
Somepalli et al., <u>"Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models",</u> CVPR 2023    
Carlini et al., <u>"Extracting Training Data from Diffusion Models",</u> arXiv 2023    
Gandikota et al., <u>"Erasing Concepts from Diffusion Models",</u> arXiv 2023    
Kumari et al., <u>"Ablating Concepts in Text-to-Image Diffusion Models",</u> arXiv 2023    
Somepalli et al., <u>"Understanding and Mitigating Copying in Diffusion Models",</u> arXiv 2023    



---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/