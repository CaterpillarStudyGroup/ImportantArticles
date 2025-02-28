P111  
# Model Adaptation  

P112    
## You’ve trained a model. What next?

![](../assets/P112图.png)    

> 已有一个预训练模，可以做什么？    

P113    
Faster Sampling   

P114    
## Faster sampling by straightening the flow   

![](../assets/P114图.png)    

$$
ℒ(θ) = \mathbb{E} _ {t,(X_0,X_1)∼π_ {0,1}^0}||u^θ_t (X_t) − (X_1 − X_0)||^2
$$

Rectified Flow refits using the **pre-trained (noise, data) coupling**.      
**Leads to straight flows**.     

“Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow” Liu et al. (2022)      
“InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation” Liu et al. (2022)    

> Rectified Flow：让 flow 从源直接到目标。     
第1步：训练 flow matching，flow matching 模型定义了源和目标的耦合关系，也得到了噪声与数据的 pair data.    
第2步：用 pair data 继续训练。      

P115     



P116    
## Faster sampling by straightening the flow    

![](../assets/P116图-1.png)    

> Result，Diffusion 对比 Rectified Flow      

$$
{\Large \mathbf{caveat} }
$$

Enforcing **straightness restricts** the model. Often a slight drop in sample quality

“InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation” Liu et al. (2022)    

P118    
## Faster sampling by self-consistency loss   

![](../assets/P118图.png)    

“One Step Diffusion via Shortcut Models” Frans et al. (2024)    

> 增大 \\(h\\)，在 \\(x_t\\) 和 \\(X_{t＋h}\\) 之间建立 shortcut，类似于 diffusion 中的蒸馏方法。     

P119    
## Faster sampling by self-consistency loss   

![](../assets/P119图.png)    

“One Step Diffusion via Shortcut Models” Frans et al. (2024)    

P121    
## Faster sampling by self-consistency loss    

![](../assets/P121图.png)    

“One Step Diffusion via Shortcut Models” Frans et al. (2024)    

> short cuts 直接预测流而不是速度，因此不能结合 CFG.       
预置 CFG 权重针对此问题的 workaround.     

P124   
## Faster sampling by only modifying the solver   

**Can adapt pre-trainedmodels to different schedulers**.  

![](../assets/P124图1.png)    

> 以上两种方法，都需训练。此方法不需要训练，而是修改 solver.     

Related by a **scaling & time** transformation:    

![](../assets/P124图2.png)    

> 补充：关于调度器．\\(\alpha _t\\) 和 \\(\sigma _t\\)．     
有一个用 scheduler A 训练好的模型，现在预要一个用 scheduler B 训练，这两个模型是什么关系？       

![](../assets/P124图3.png)    

> 结论：这两个 scheduler 及其 flow 可以通过 \\(X\\) 的缩放和时间的重参数化关联起来。       
时间重参数化是指，调整 scheduler 的 SNR.     

“Elucidating the design space of diffusion-based generative models” Karras et al. (2023)    
“Bespoke Solvers for Generative Flow Models” Shaul et al. (2023)     

P126   
## Faster sampling by only modifying the solver   

**Bespoke solvers:**    
**Decouples** model & solver.     
Model is left unchanged.    
Parameterize solver and optimize.   

**Can be interpreted as** finding best scheduler + more.   

**Solver consistency:** sample quality is retained as NFE → ∞.    

![](../assets/P126图.png)    

“Bespoke Solvers for Generative Flow Models” Shaul et al. (2023)    
“Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models” Shaul et al. (2024)    

> Bespoke Solver：      
1．模型与 solver 解耦，模型不变，仅优化求 solver.      
2．向 solver 中仅入参数(表达 scheduler)，优化这些参数。     
由于仅优化solver，好处：    
1．可以保持 solver 的一致性。     
2．在不同的模型(不同数据集、分辨率等训练出来的模型)之间可迁移。     
局限性：    
虽然能(不重训)直接迁移到另一个模型，但比在另一个模型上蒸馏(重训)效果要差一点。       

P127    
## Faster sampling by only modifying the solver

Bespoke solvers can t**ransfer across different data sets and resolutions**.     

$$
\mathrm{Caveat} 
$$

However, **does not reach distillation performance at extremely low NFEs.**    

“Bespoke Solvers for Generative Flow Models” Shaul et al. (2023)     
“Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models” Shaul et al. (2024)     

P128   
## Faster sampling references   

**Rectified flows:**    
“Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow” Liu et al. (2022)     
“InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation” Liu et al. (2024)     
“Improving the Training of Rectified Flows” Lee et al. (2024)    

**Consistency & shortcut models:**      
“Consistency Models” Song et al. (2023)    
“Improved Techniques for Training Consistency Models” Song & Dhariwal (2023)    
“One Step Diffusion via Shortcut Models” Frans et al. (2024)    

**Trained & bespoke solvers:**    

“DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics” Zheng et al. (2023)     
“Bespoke Solvers for Generative Flow Models” Shaul et al. (2023)     
“Bespoke Non-Stationary Solvers for Fast Sampling of Diffusion and Flow Models” Shaul et al. (2024)     

P129   
## You’ve trained a model. What next?   

Inverse Problems (Training-Free)     

> Inverse Problem：填充、去糊、超分、编辑。       

P133    
## Solving inverse problems by posterior inference   

![](../assets/P133图.png)    

“Pseudoinverse-Guided Diffusion Models for Inverse Problems” Song et al. (2023)    
“Training-free Linear Image Inverses via Flows” Pokle et al. (2024)    

> \\(x_1\\) 为干净图像，\\(y\\) 为噪声图像。    
用高斯来近似其中未知的部分 (score function)      
score function 可能是 multi 的，但实验证明仅用高斯也能有比较好的效果。     

P135    

> 预训练一个生成模型，然后有这个模型来评估数据，评估结果很不可靠，它把真实数据评估为高密度，非真实数据评估为低密度。       
因为，高密度\\(\ne\\) 高采样率。     

P138     

> 逆问题转化为优化问题。     

$$
X_1=\psi (X_0)
$$

\\(\psi \\) 是预训练的生成模型，不优化 \\(\psi \\) 的参数，那就优化\\(X_0\\) 因为 \\(\psi \\) 是一个平滑、可逆、可微的函数。     
