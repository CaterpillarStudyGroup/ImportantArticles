
P1   
## Denoising Diffusion Models A Generative Learning Big Bang 

##### CVPR 2023 Tutorial

Part III: Applications on other domains    

P2   
## Outline    
 - Inverse problems
 - Setup
 - Replacement-based methods
 - Reconstruction-based methods

P3   
## Diffusion Models for Inverse Problems

Goal: denoise and super-resolve an image    

![](../assets/D3-3.png)  

![](../assets/D3-3-1.png)  

P4   
## Diffusion Models for Inverse Problems

Goal: recover the masked region of an image

![](../assets/D3-4-2.png)  

![](../assets/D3-4.png)  

P5   
## Diffusion Models for Inverse Problems: Two Paradigms

![](../assets/D3-5-1.png)  

**Replacement-based methods**    
(Overwrites model prediction with known information)    

![](../assets/D3-5-2.png)  

**Reconstruction-based methods**    
(Approximate classifier-free guidance **without additional training**)    

P6   
## Replacement-based Methods: An Example   

![](../assets/D3-6.png)  

Song et al., <u>"Score-Based Generative Modeling through Stochastic Differential Equations",</u> ICLR 2021    

P7   
## Reconstruction-based Methods: An Example

![](../assets/D3-7.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023   

P8    
## Diffusion Posterior Sampling   

In the Gaussian case,    

$$
p(\mathbf{y} |\mathbb{E} [\mathbf{x} _0|\mathbf{x} _t])=-c||\mathcal{A} \mathbf{(\hat{x}}  _0)-\mathbf{y} ||^2_2
$$

Maximizing the likelihood is minimizing the L2 distance between measured and generated!     

![](../assets/D3-8.png)  

Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> ICLR 2023    


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
## Outline

 - **3D**    
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

P13    
## Diffusion Models for Point Clouds    

![](../assets/D3-13.png)  

Zhou et al., <u>"3D Shape Generation and Completion through Point-Voxel Diffusion",</u> ICCV 2021    

P14     
## Diffusion Models for Point Clouds   

![](../assets/D3-14.png)  

Zeng et al., <u>"LION: Latent Point Diffusion Models for 3D Shape Generation",</u> NeurIPS 2022    

P15   
## Diffusion Models for Point Clouds

Point-E uses a synthetic view from fine-tuned GLIDE, and then ”lifts” the image to a 3d point cloud.

![](../assets/D3-15.png)  

Nichol et al., <u>"Point-E: A System for Generating 3D Point Clouds from Complex Prompts",</u> arXiv 2022     

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

P18   
## Diffusion Models for Signed Distance Functions

![](../assets/D3-18.png)  

Latent space diffusion for SDFs, where conditioning can be provided with cross attention

Chou et al., <u>"DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions",</u> arXiv 2022    

P19   
## Diffusion Models for Other 3D Representations    

Neural Radiance Fields (NeRF) is another representation of a 3D object.    

![](../assets/D3-19.png)  
 
P20   
## Diffusion Models for Other 3D Representations

![](../assets/D3-20-1.png)  
NeRF    
(Fully implicit)    

![](../assets/D3-20-2.png)  
Voxels    
(Explicit / hybrid)    

![](../assets/D3-20-3.png)  
Triplanes    
(Factorized, hybrid)    

Image from EG3D paper.    

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

P22    
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

P26   
## DreamFusion: Score Distillation Sampling   

![](../assets/D3-26.png)     

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023     


P27   
## DreamFusion: Score Distillation Sampling

Consider the KL term to minimize (given t):   

$$
\mathbf{KL} (q(\mathbf{z} _t|g(\theta );y,t)||p\phi (\mathbf{z} _t;y,t))
$$

KL between noisy real image distribution and generated image 
distributions, conditioned on y!     

KL and its gradient is defined as:    

![](../assets/D3-27.png)  

(B) can be derived from chain rule    

$$
\nabla _\theta \log p_\phi (\mathbf{z} _t|y)=s_\phi (\mathbf{z} _t|y)\frac{\partial \mathbf{z} _t}{\partial \theta }=\alpha _ts_\phi (\mathbf{z} _t|y)\frac{\partial \mathbf{x} }{\partial \theta } =-\frac{\alpha _t}{\sigma _t}\hat{\epsilon }_ \phi (\mathbf{z} _t|y)\frac{\partial \mathbf{x} }{\partial \theta }   
$$

(A) is the gradient of the entropy of the forward process with fixed variance = 0.    

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023   

P28    
## DreamFusion: Score Distillation Sampling  

$$
(A)+(B)=\frac{\alpha _t}{\sigma _t}\hat{\epsilon }_ \phi (\mathbf{z} _t|y)\frac{\partial \mathbf{x} }{\partial \theta }
$$

However, this objective can be quite noisy.     
Alternatively, we can consider a “baseline” approach in reinforcement learning: add a component that has zero mean but reduces variance. Writing out (A) again:     

![](../assets/D3-28-1.png)  

Thus, we have:

![](../assets/D3-28-2.png)  

This has the same mean, but **reduced variance**, as we train \\(\hat{\epsilon } _\phi\\) to predict \\(\epsilon\\)    


Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023    

P29   
## DreamFusion in Text-to-3D    

 - SDS can be used to optimize a 3D representation, like NeRF.   

![](../assets/D3-29.png)  

Poole et al., <u>"DreamFusion: Text-to-3D using 2D Diffusion",</u> ICLR 2023    

P30   
## Extensions to SDS: Magic3D

2x speed and higher resolution   
 - Accelerate NeRF with Instant-NGP, for coarse representations.    
 - Optimize a fine mesh model with differentiable renderer.   

![](../assets/D3-30.png)  

Lin et al., <u>"Magic3D: High-Resolution Text-to-3D Content Creation",</u> CVPR 2023   

P31
## Alternative to SDS: Score Jacobian Chaining

A different formulation, motivated from approximating 3D score.   

![](../assets/D3-31.png)  

In principle, the diffusion model is the noisy 2D score (over clean images),   
but in practice, the diffusion model suffers from out-of-distribution (OOD) issues!    

For diffusion model on noisy images, **the non-noisy images are OOD**!    

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.     

P32   
## Score Jacobian Chaining   

SJC approximates noisy score with “Perturb-and-Average Scoring”, which is not present in SDS.   
 - Use score model on multiple noise-perturbed data, then average it.    

![](../assets/D3-32.png)  

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.    

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
\min_{\mu } D_{\mathrm{KL} }(q^\mu _0(x_0|y)||p_0(x_0|y)). \\\\
\quad \mu \quad \text{is the distribution of NeRFs} .
\end{matrix}
$$

Suppose is a \\(\theta _\tau \sim \mu \\) NeRF sample, then VSD simulates this ODE:    

![](../assets/D3-35.png) 

 - Diffusion model can be used to approximate score of noisy real images.   
 - How about noisy rendered images?   sss

P36   
## ProlificDreamer and Variational Score Distillation

 - Learn another diffusion model to approximate the score of noisy rendered images!

![](../assets/D3-36.png) 

Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

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

P41    
## GenVS   

 - 3D-aware architecture with latent feature field.    
 - Use diffusion model to improve render quality based on structure.   

![](../assets/D3-41.png)  

Chan et al., <u>"Generative Novel View Synthesis with 3D-Aware Diffusion Models",</u> arXiv 2023    

P42    
## Outline   

 - 3D reconstruction    

P43    
## NeuralLift-360 for 3D reconstruction

 - SDS + Fine-tuned CLIP text embedding + Depth supervision    

![](../assets/D3-43.png)  

Xu et al., <u>"NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views",</u> CVPR 2023    

P44    
## Zero 1-to-3   

 - Generate novel view from 1 view and pose, with 2d model.    
 - Then, run SJC / SDS-like optimizations with view-conditioned model.   

![](../assets/D3-44.png)  

Liu et al., <u>"Zero-1-to-3: Zero-shot One Image to 3D Object",</u> arXiv 2023    

P45    
## Outline

 - 3D editing

P46   
## Instruct NeRF2NeRF

Edit a 3D scene with text instructions   

![](../assets/D3-46.png)  

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023      

P47   
## Instruct NeRF2NeRF   

**Edit a 3D scene with text instructions**   
 -  Given existing scene, use Instruct Pix2Pix to edit image at different viewpoints.   
 - Continue to train the NeRF and repeat the above process   

![](../assets/D3-47.png)  

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023   






![](../assets/D3-72.png)  

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/