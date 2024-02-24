
P38   
# Accelerated Sampling    

P39   
## What makes a good generative model?    
##### The generative learning trilemma

![](../../assets/D1-39.png) 

<u>Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, ICLR 2022</u>    

P40   
## What makes a good generative model?    
##### The generative learning trilemma   

Tackle the trilemma by accelerating diffusion models    

<u>Tackling the Generative Learning Trilemma with Denoising Diffusion GANs, ICLR 2022</u>   


P41  
## Acceleration Techniques   

Advanced ODE/SDE Solvers    
Distillation Techniques    
Low-dim. Diffusion Processes     
Advanced Diffusion Processes    

P42   

> &#x2705; ODE 实现 std normal 分布与真实数据分布之间的映射。    


P43   
## Generative ODEs    
##### Solve ODEs with as little function evaluations as possible    

$$
dx=\epsilon _\theta (x,t)dt
$$

![](../../assets/D1-43.png) 


> &#x2705; Euler 方法：每个时间步简化为线性过程。当 step 较大时，会与 GT 有较大的偏离。     


P44    

![](../../assets/D1-44.png) 

Song et al., <u>"Denoising Diffusion Implicit Models (DDIM)",</u> ICLR 2021    

P45   
![](../../assets/D1-45.png) 

P46   
## A Rich Body of Work on ODE/SDE Solvers for Diffusion Models

 - Runge-Kutta adaptive step-size ODE solver:   
    - <u>Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations”, *ICLR*, 2021</u>   
 - Higher-Order adaptive step-size SDE solver:    
    - <u>Jolicoeur-Martineau et al., “Gotta Go Fast When Generating Data with Score-Based Models”, *arXiv*, 2021</u>    
 - Reparametrized, smoother ODE:   
    - <u>Song et al., “Denoising Diffusion Implicit Models”, *ICLR*, 2021</u>   
    - <u>Zhang et al., "gDDIM: Generalized denoising diffusion implicit models", arXiv 2022</u>   
 - Higher-Order ODE solver with linear multistepping:   
    - <u>Liu et al., “Pseudo Numerical Methods for Diffusion Models on Manifolds”, *ICLR*, 2022</u>   
 - Exponential ODE Integrators:   
    - <u>Zhang and Chen, “Fast Sampling of Diffusion Models with Exponential Integrator”, *arXiv*, 2022</u>   
    - <u>Lu et al., “DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps”, *NeurIPS*, 2022</u>   
    - <u>Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models", NeurIPS 2022</u>   
 - Higher-Order ODE solver with Heun’s Method:   
    - <u>Karras et al., “Elucidating the Design Space of Diffusion-Based Generative Models”, *NeurIPS*, 2022</u>   
 - Many more:   
    - <u>Zhao et al., "UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models", arXiv 2023</u>    
    - <u>Shih et al., "Parallel Sampling of Diffusion Models", arxiv 2023</u>     
    - <u>Chen et al., "A Geometric Perspective on Diffusion Models", arXiv 2023</u>     

P48    
## ODE Distillation

![](../../assets/D1-48.png) 

Can we train a neural network to directly predict \\(\mathbf{x} _{{t}'} \\) given \\(\mathbf{x} _t\\)?    

P49    

> &#x1F50E; Salimans & Ho, “**Progressive distillation** for fast sampling of diffusion models”, ICLR 2022. [link](https://caterpillarstudygroup.github.io/ReadPapers/1.html)


> &#x1F50E; Meng et al., "On Distillation of Guided Diffusion Models", CVPR 2023 (**Award Candidate**). [link](https://caterpillarstudygroup.github.io/ReadPapers/3.html)



P51    

> &#x1F50E; Song et al., **Consistency Models**, ICML 2023 [link](https://caterpillarstudygroup.github.io/ReadPapers/7.html)


P52   
## SDE Distillation

![](../../assets/D1-52.png) 

Can we train a neural network to directly predict **distribution of** \\(\mathbf{x} _ {{t}'} \\) given \\(\mathbf{x} _ t \\) ?    

> &#x2705; \\(\mathbf{x} _ t\\) 与 \\( \mathbf{x} _ {{t}' }\\) 没有必然的联系，得到的是 \\( \mathbf{x} _ {{t}' }\\) 的分布。    

P53   
## Advanced Approximation of Reverse Process    
##### Normal assumption in denoising distribution holds only for small step    

![](../../assets/D1-53.png) 

**Requires more complicated functional approximators!**   
GANs used by Xiao et al.    
Energy-based models by Gao et al.    


<u>Xiao et al., “Tackling the Generative Learning Trilemma with Denoising Diffusion GANs”, ICLR 2022.</u>    
<u>Gao et al., “Learning energy-based models by diffusion recovery likelihood”, ICLR 2021.</u>    


> &#x2705; 从 \\(t\\) 与 \\({t}'\\) 的差距过大时，normal 分布不足以表达 \\(q(\mathbf{x} _ {{t}'}｜\mathbf{x} _ t)\\).    


P54   
## Training-based Sampling Techniques

 - Knowledge distillation:   
    - Luhman and Luhman, <u>Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed,</u> arXiv 2021    
 - Learned Samplers:   
    - Watson et al., <u>"Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality",</u> ICLR 2022    
 - Neural Operators:   
    - Zheng et al., <u>Fast Sampling of Diffusion Models via Operator Learning, </u>ICML 2023
 - Wavelet Diffusion Models:    
    - Phung et al., <u>"Wavelet Diffusion Models Are Fast and Scalable Image Generators", </u> CVPR 2023    
 - Distilled ODE Solvers:   
    - Dockhorn et al., <u>"GENIE: Higher-Order Denoising Diffusion Solvers",</u> NeurIPS 2022    

P56   
## Cascaded Generation    
##### Pipeline    

![](../../assets/D1-56.png) 

Cascaded Diffusion Models outperform Big-GAN in FID and IS and VQ-VAE2 in Classification Accuracy Score.    

<u>Ho et al., “Cascaded Diffusion Models for High Fidelity Image Generation”, 2021.</u>     
<u>Ramesh et al., “Hierarchical Text-Conditional Image Generation with CLIP Latents”, arXiv 2022.</u>     
<u>Saharia et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding”, arXiv 2022.</u>     


P57   
## Latent Diffusion Models   
##### Variational autoencoder + score-based prior   

![](../../assets/D1-57.png) 

##### Main Idea   

Encoder maps the input data to an embedding space    
Denoising diffusion models are applied in the latent space    

<u>Vahdat et al., “Score-based generative modeling in latent space”, NeurIPS 2021.</u>    

P58   
##### Advantages:    
(1) The distribution of latent embeddings close to Normal distribution \\(\to \\) ***Simpler denoising, Faster synthesis***!    
(2) Latent space \\(\to \\) ***More expressivity and flexibility in design!***    
(3) Tailored Autoencoders \\(\to \\) ***More expressivity, Application to any data type (graphs, text, 3D data, etc.)!***     

<u>Vahdat et al., “Score-based generative modeling in latent space”, NeurIPS 2021.</u>    

P59
## Latent Diffusion Models   
##### End-to-End Training objective    

![](../../assets/D1-59.png) 

<u>Vahdat et al., “Score-based generative modeling in latent space”, NeurIPS 2021.</u>   

> &#x2705; 这篇文章对 VAE 和 diffusion 一起进行训练，文章的创新点是，利用 score matching 中的信息来计算 cross entropy.    


P60   
## Latent Diffusion Models    
##### Two-stage Training   

The seminal work from Rombach et al. CVPR 2022:    
 - Two stage training: train autoencoder first, then train the diffusion prior   
 - Focus on compression without of any loss in reconstruction quality   
 - Demonstrated the expressivity of latent diffusion models on many conditional problems    

The efficiency and expressivity of latent diffusion models + open-source access fueled a large body of work in the community    

<u>Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models”, CVPR 2022.</u>     

> &#x2705; 文章特点：1. VAE 和 diffusion 分开训练    
> &#x2705; 2. 使用的数据集质量比较高   
> &#x2705; 3. Advanced Auto Encoders     
> &#x2705; 4. Adveseral Training

P61    
## Additional Reading    

More on low-dimensional diffusion models:    
 - Sinha et al., <u>"D2C: Diffusion-Denoising Models for Few-shot Conditional Generation", </u> NeurIPS 2021    
 - Daras et al., <u>"Score-Guided Intermediate Layer Optimization: Fast Langevin Mixing for Inverse Problems",</u> ICML 2022    
 - Zhang et al., <u>“Dimensionality-Varying Diffusion Process”, </u>arXiv 2022.    


P63   
## ODE interpretation    
##### Deterministic generative process   

![](../../assets/D1-63.png) 

 - DDIM sampler can be considered as an integration rule of the following ODE:    

$$
d\mathbf{\bar{x} } (t)=\epsilon ^{(t)} _ \theta(\frac{\mathbf{\bar{x} } (t)}{\sqrt{\eta ^2+1}} )d\eta (t); \mathbf{\bar{x} } =\mathbf{x} / \sqrt{\bar{a} },\eta = \sqrt{1-\bar{a}} / \sqrt{\bar{a } }
$$

 - Karras et al. argue that the ODE of DDIM is favored, as the tangent of the solution trajectory always points 
towards the denoiser output.   

 - This leads to largely linear solution trajectories with low curvature à Low curvature means less truncation 
errors accumulated over the trajectories. 

<u>Song et al., “Denoising Diffusion Implicit Models”, ICLR 2021.</u>   
<u>Karras et al., “Elucidating the Design Space of Diffusion-Based Generative Models”, arXiv 2022.</u>   
<u>Salimans & Ho, “Progressive distillation for fast sampling of diffusion models”, ICLR 2022.</u>   

P64   
## “Momentum-based” diffusion      

##### Introduce a velocity variable and run diffusion in extended space

![](../../assets/D1-64.png) 

<u>Dockhorn et al., “Score-Based Generative Modeling with Critically-Damped Langevin Diffusion”, ICLR 2022.</u>     

P65   
## Additional Reading

 - Schrödinger Bridge:    
    - Bortoli et al., <u>"Diffusion Schrödinger Bridge",</u> NeurIPS 2021    
    - Chen et al., <u>“Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory”, </u>ICLR 2022    
 - Diffusion Processes on Manifolds:   
    - Bortoli et al., <u>"Riemannian Score-Based Generative Modelling", </u>NeurIPS 2022    
 - Cold Diffusion:    
    - Bansal et al., <u>"Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise", </u>arXiv 2022      
 - Diffusion for Corrupted Data:    
    - Daras et al., <u>"Soft Diffusion: Score Matching for General Corruptions", </u>TMLR 2023      
   - Delbracio and Milanfar, <u>"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration", </u>arXiv 2023    
   - Luo et al., <u>"Image Restoration with Mean-Reverting Stochastic Differential Equations", </u>ICML 2023    
   - Liu et al., <u>“I2SB: Image-to-Image Schrödinger Bridge”, </u>ICML 2023    
 - Blurring Diffusion Process:    
    - Hoogeboom and Salimans, <u>"Blurring Diffusion Models", </u>ICLR 2023   
   - Rissanen et al, <u>“Generative Modelling With Inverse Heat Dissipation”, </u>ICLR 2023  



---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/