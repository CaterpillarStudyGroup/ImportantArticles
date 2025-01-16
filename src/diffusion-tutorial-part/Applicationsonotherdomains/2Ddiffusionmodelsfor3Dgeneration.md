P23   
## 2D Diffusion Models for 3D Generation   
Just now, we discussed diffusion models directly on 3d.   
However, there are a lot fewer 3d data than 2d.A lot of experiments are based on ShapeNet!   
Can we **use 2d diffusion models as a “prior” for 3d**?   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|**68**|2023|Poole et al., "DreamFusion: Text-to-3D using 2D Diffusion"||SDS|[link](https://caterpillarstudygroup.github.io/ReadPapers/68.html)|
||2023|Lin et al., <u>"Magic3D: High-Resolution Text-to-3D Content Creation"|2x speed and higher resolution <br>Accelerate NeRF with Instant-NGP, for coarse representations. <br> Optimize a fine mesh model with differentiable renderer.<br> &#x2705; Instant NGP 代替左下的 Nerf MLP．以 coarse representetion 作为 condition 来生成 fine mesh model.  |Extensions to SDS<br>![](../../assets/D3-30.png)  |
||2023|Wang et al.,"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",||  Alternative to SDS|


P31
## Alternative to SDS: Score Jacobian Chaining

A different formulation, motivated from approximating 3D score.   

![](../../assets/D3-31.png)  

In principle, the diffusion model is the noisy 2D score (over clean images),   
but in practice, the diffusion model suffers from out-of-distribution (OOD) issues!    

For diffusion model on noisy images, **the non-noisy images are OOD**!    

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.     

> &#x2705; 2D sample, 3D score    

P32   
### Score Jacobian Chaining   

SJC approximates noisy score with “Perturb-and-Average Scoring”, which is not present in SDS.   
 - Use score model on multiple noise-perturbed data, then average it.    

![](../../assets/D3-32.png)  

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.    

> &#x2705; 通过这种方法来近似 clean image 的输出，解决 clean image 的 OOD 问题。    


P33    
### SJC and SDS

SJC is a competitive alternative to SDS.   

![](../../assets/D3-33.png) 

Wang et al., <u>"Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation",</u> CVPR 2023.    

P34   
## Alternative to SDS: ProlificDreamer   

 - SDS-based method often set classifier-guidance weight to 100, which limits the “diversity” of the generated samples.   
 - ProlificDreamer reduces this to 7.5, leading to diverse samples.    

![](../../assets/D3-34.png) 

Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

P35   
### ProlificDreamer and Variational Score Distillation  

Instead of maximizing the likelihood under diffusion model, VSD minimizes the KL divergence via variational inference.    

$$
\begin{matrix}
\min_{\mu } D _ {\mathrm{KL} }(q^\mu _ 0(\mathbf{x} _ 0|y)||p _ 0(\mathbf{x} _ 0|y)). \\\\
\quad \mu \quad \text{is the distribution of NeRFs} .
\end{matrix}
$$

Suppose is a \\(\theta _ \tau \sim \mu \\) NeRF sample, then VSD simulates this ODE:    

![](../../assets/D3-35.png) 

 - Diffusion model can be used to approximate score of noisy real images.   
 - How about noisy rendered images?   sss

> &#x2705; 第一项由 diffusion model 得到，在此处当作 GT．   

P36   
### ProlificDreamer and Variational Score Distillation

 - Learn another diffusion model to approximate the score of noisy rendered images!

![](../../assets/D3-36.png) 

Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    

> &#x2705; 使用 LoRA 近第二项。    

P37   
### Why does VSD work in practice?    

 - The valid text-to-image NeRFs form a distribution with infinite possibilities!    
 - In SDS, epsilon is the score of noisy “dirac distribution” over finite renders, which converges to the true score with infinite renders!    
 - In VSD, the LoRA model aims to **represent the (true) score of noisy distribution over infinite number of renders!**   
 - If the generated NeRF distribution is only one point and LoRA overfits perfectly, then VSD = SDS!    
 - But LoRA has good generalization (and learns from a trajectory of NeRFs), so closer to the true score!    

 - This is analogous to    
    - Representing the dataset score via mixture of Gaussians on the dataset (SDS), versus     
    - Representing the dataset score via the LoRA UNet (VSD)    


Wang et al., <u>"ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation",</u> arXiv 2023    



---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/