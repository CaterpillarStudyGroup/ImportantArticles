
P34     
# Flow Matching Advanced Designs


P35     

![](../assets/P35图.png)    

> 1．条件生成    
2．\\(P\\) 分布和 \\(Q\\) 分布耦合的场景    
3．在几何域上使用 flow matching 构造生成模型    

P37     

## Conditioning and Guidance 

![](../assets/P37图.png)    


P39     

## Conditional Models

$$
p_ {t,1|Y} (x, x_1|y) = p_ {t|1}(x|x_1)q(x_1|y)
$$

![](../assets/P39图1.png)    

![](../assets/P39图2.png)  

> 将条件概率路径构建为不显式依赖于条件 \\(Y\\)。     


P40    
## Conditional Models

Train same neural network on all conditions:     

![](../assets/P40图1.png)    

![](../assets/P40图2.png)  

> 对于网络训练的影响在于，数据增加一个维度来表示\\(Y\\)。     


P41    
## Conditional Models - Examples

![](../assets/P41图.png)    

“Flow Matching for Generative Modeling” Lipman et al. (2022)       
“GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models” Nichol et al. (2021)     

> 此方法在“每个条件都有大量数据”时很有用，例如条件是类别时。      
条件是文本时不适用，因为数据集里一段文本通常只对应一张图像。     

P42     
## Guidance for Score Matching

![](../assets/P42图.png)    

> classifier Guidance：通过引入分类器，将无条件模型变成条件模型.     
CFG：条件生成结果与无条件生成结果外插。    

P43     
## Guidance for Flow Matching    

Assume a velocity field trained with **Gaussian paths**.      

![](../assets/P43图.png)    

> 以上是来自 score matching 的公式，同样的方法适配到 flow matching.    

P44    
Flow Matching with Classifier-Free guidance:     
"Guided Flows for Generative Modeling and Decision Making" Zheng et al. (2023)     
"Mosaic-SDF for 3D Generative Models" Yariv et al. (2023)    
"Audiobox: Unified Audio Generation with Natural Language Prompts" Vyas et al. (2023)        
"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" Esser et al. (2024)       
"Movie Gen: A Cast of Media Foundation Models" Polyak et al. (2024)     


P45   
## Guidance for Flow Matching

"Movie Gen: A Cast of Media Foundation Models" Polyak et al. (2024)     
![](../assets/P45图.png)    

> 基于 CFG 训练的 flow matching，在生成质量和文本一致性上，均优于 diffusion.     

P46    
## Guidance for Flow Matching

**Open Problem**     
How to guide FM with non-Gaussian paths?      

> CFG 要求正在学习时二
模型、但 flow matching 不局限于高斯源。     

P47    
Data Couplings     

![](../assets/P47图.png)    

P49    

$$
X_t = ψ_t(X_0|X_1)
$$

![](../assets/P49图.png) 

$$
(X_0,X_1) ∼ π_{0,1} = p(X_0)q(X_1)
$$

What about dependent couplings?      

> 前面工作都假设 \\(P\\) 和 \\(Q\\) 是独立的。     

P50    

> 两种方法，利用 \\(P\\) 和 \\(Q\\) 的耦合关系优化生成过程。    
1．利用耦合关系，构造另一种条件方法。     
2．试图找到多样本之间的耦合关系。      

P52    
## Data Couplings

|||
|--|--|
| ![](../assets/P52图1.png)  | ![](../assets/P52图2.png)  |
| • Non-Gaussian source distribution <br>• Alternative conditioning approach <br>• Inverse problems | • Applications to Optimal Transport <br> • Efficiency: straighter trajectories |


P58    
## Paired Data

![](../assets/P58图1.png)     

![](../assets/P58图2.png)     

Alter **source distribution** and **coupling** instead of adding **condition**      

"I2SB: Image-to-Image Schrödinger Bridge" Liu et al. (2023)    
"Stochastic interpolants with data-dependent couplings" Albergo et al. (2024)    

> 改变源分布和耦合，而不是添加条件。    
源分布不是噪声，而是 \\(Y\\) 添加噪声，损失不变。     

P60    

![](../assets/P60图.png)     

P61    
## Result   

![](../assets/P61图.png)     

"Stochastic interpolants with data-dependent couplings" Albergo et al. (2024)   

P63    
## Multisample Couplings   

![](../assets/P63图.png)     

Given uncoupled **source** and **target** distributions,can we build a coupling to induce straighter paths?      

"Multisample Flow Matching: Straightening Flows with Minibatch Couplings" Pooladian et al. (2023)    
"Improving and generalizing flow-based generative models with minibatch optimal transport" Tong et al. (2023)     

> 有一个预训练的 flow matching 模型，构建一种耦合，使 \\(P\\) 到 \\(Q\\) 的路径更直线，或 \\(Q\\) 能更好地采样。    

P64    
## Multisample Couplings   

![](../assets/P64图.png)    

Marginal \\(u_t\\) with cond-OT FM and \\(π_{0,1}\\)      

"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" Liu et al. (2022)     
"Multisample Flow Matching: Straightening Flows with Minibatch Couplings" Pooladian et al. (2023)      
"Improving and generalizing flow-based generative models with minibatch optimal transport" Tong et al. (2023)      

> 耦合 cost 限制了动能．降低 coupling cost，就能减少动能。     

P69   
## Multisample Couplings    

Use mini batch optimal transport couplings     

![](../assets/P69图.png)    

"Multisample Flow Matching: Straightening Flows with Minibatch Couplings" Pooladian et al. (2023)     
"Improving and generalizing flow-based generative models with minibatch optimal transport" Tong et al. (2023)    

> 从 \\(P\\) 分布和 \\(Q\\) 分布中随机采样 \\(k\\) 个点，寻找两组点之间的最优排列，来最小化 cost.    
假设找到了最优组合，随机选择一对。    

P70    
$$
\mathrm{When} \quad k = 1 → π_{0,1} = p(X_0)q(X_1)
$$

> 当 \\(k＝1\\) 时，相当于 \\(P\\) 和 \\(Q\\) 是独立的。    


P71    
**FM with cond-OT is not marginal OT!**    

When \\(k → ∞, u_t\\) generates the OT map    

P72    
## Multisample Couplings    

![](../assets/P72图.png)    

"SE(3)-Stochastic Flow Matching for Protein Backbone Generation" Bose et al. (2023)     
"Multisample Flow Matching: Straightening Flows with Minibatch Couplings" Pooladian et al. (2023)     
"Improving and generalizing flow-based generative models with minibatch optimal transport" Tong et al. (2023)    

> 低维时，此方法能明显降低 cost；高维时，路径本身已接近直线，因此效果不明显。     

P73    
## Data Couplings    

**Paired data:**     
"I2SB: Image-to-Image Schrödinger Bridge" Liu et al. (2023)    
"Stochastic interpolants with data-dependent couplings" Albergo et al. (2024)    
"Simulation-Free Training of Neural ODEs on Paired Data" Kim et al. (2024)    

**Multisample couplings:**     
"Multisample Flow Matching: Straightening Flows with Minibatch Couplings" Pooladian et al. (2023)     
"Improving and generalizing flow-based generative models with minibatch optimal transport" Tong et al. (2023)    
"SE(3)-Stochastic Flow Matching for Protein Backbone Generation" Bose et al. (2023)   
"Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Backbone Generation" Huguet et al. (2024)   


P75    
## Geometric Flow Matching

|||
|--|--|
| Data with Symmetries | Riemannian Manifolds |
| ![](../assets/P75图1.png) | ![](../assets/P75图3.png) |
| • Equivariant flows → invariant densities <br>• Alignment couplings | • Simulation free on simple manifolds <br> • General geometries |

P87    
## Equivariant Flows   

![](../assets/P87图1.png)     

![](../assets/P87图2.png)     

"Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities" Köhler et al. (2020)     

> 有些对象具有对称性，希望生成的对象也能满足这些特算。     

P88    
## Equivariant Flow Matching    

Equivariant Velocity     

$$ 
u^θ_t (g⋅x) = g⋅u^θ_t(x)
$$

Train with CFM:     
  
![](../assets/P88图2.png)     

"Equivariant flow matching" Klein et al. (2023)     
"Equivariant Flow Matching with Hybrid Probability Transport" Song et al. (2023)    

P89    

![](../assets/P89图.png)     

> 假设 \\(P\\) 和 \\(Q\\) 是独立的，会发生这种情况。    

P90    
## Equivariant Flow Matching    

Equivariant Velocity     

$$ 
u^θ_t (g⋅x) = g⋅u^θ_t(x)
$$

Train with CFM:    

![](../assets/P90图1.png)     

![](../assets/P90图2.png)    

> 导致模型学到的轨迹弯曲。     

P91   

"Equivariant flow matching" Klein et al. (2023)    
"Equivariant Flow Matching with Hybrid Probability Transport" Song et al. (2023)    

P92     
## Equivariant Flow Matching    

|||
|--|--|
| "Fast Point Cloud Generation with Straight Flows" Wu et al. (2022) | "Equivariant Flow Matching with Hybrid Probability Transport" Song et al. (2023)<br> "Equivariant flow matching" Klein et al. (2023)|
| ![](../assets/P92图1.png)  | ![](../assets/P92图2.png) |

P94    
## Generative Modeling on Manifolds    


P95   
Need to re-define the geometric structures we have in Euclidean space.     

P98    
## Riemannian Manifolds

![](../assets/P98图.png)    

P99   

![](../assets/P99图.png)    


P101   
## Riemannian Flow Matching

• **Riemannian Flow Matching loss:**   

![](../assets/P101图.png)    

"Flow Matching on General Geometries" Chen & Lipman (2023)   

P102    
• **Riemannian Conditional Flow Matching loss:**     

![](../assets/P102图.png)   

Theorem: Losses are equivalent,     

$$
∇_θℒ_{RFM}(θ) = ∇_θℒ_{RCFM}(θ)
$$

"Flow Matching on General Geometries" Chen & Lipman (2023)    

P103   
## Conditional Flows - Simple Geometries    

![](../assets/P103图1.png)   

![](../assets/P103图2.png)   

**For simple manifolds** (e.g. Euclidean, sphere, torus, hyperbolic):   

![](../assets/P103图3.png)   

"Flow Matching on General Geometries" Chen & Lipman (2023)   

P104    
## Conditional Flows - General Geometries    

**Geodesics** can be hard to compute    

![](../assets/P104图.png)   

Concentrate probability at boundary    

"Flow Matching on General Geometries" Chen & Lipman (2023)   

P105   
## Conditional Flows - General Geometries   

Choose a **premetric** satisfying:     
1. Non-negative:\\(d(x,y) ≥ 0\\).      
2. Positive: \\(d(x, y) = 0\\) iff \\(x = y\\).     
3. Non-degenerate:\\(∇d(x, y) ≠ 0\\) iff \\(x ≠ y\\).    

Build **conditional flow** satisfying:    

$$ 
d(ψ_t(x_0|x_1),x_1) = \tilde{κ}(t)d(x_0,x_1)
$$

$$
\mathrm{Scheduler}  \quad \tilde{κ} (t) = 1 − κ(t)
$$

"Flow Matching on General Geometries" Chen & Lipman (2023)   

P106   

![](../assets/P106图.png)   

"Flow Matching on General Geometries" Chen & Lipman (2023)    
 
P107   

![](../assets/P107图.png)   


P108   

![](../assets/P108图.png)   

"Riemannian Score-Based Generative Modelling" De Bortoli et al. (2022)    
"Flow Matching on General Geometries" Chen & Lipman (2023)      

P109    
## Riemannian Flow vs. Score Matching

![](../assets/P109图1.png)   

![](../assets/P109图2.png)   

"Riemannian Score-Based Generative Modelling" De Bortoli et al. (2022)     
"Flow Matching on General Geometries" Chen & Lipman (2023)    

P110    
## Geometric Flow Matching   

**Equivariant Flow Matching:**    

"Fast Point Cloud Generation with Straight Flows" Wu et al. (2022)    
"Equivariant flow matching" Klein et al. (2023)    
"Equivariant Flow Matching with Hybrid Probability Transport" Song et al. (2023)     
"Mosaic-SDF for 3D Generative Models" Yariv et al. (2023)    

**Riemannian Flow Matching:**     

"Flow Matching on General Geometries" Chen & Lipman (2023)     
"SE(3)-Stochastic Flow Matching for Protein Backbone Generation" Bose et al. (2023)    
"Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Backbone Generation" Huguet et al. (2024)    
"FlowMM: Generating Materials with Riemannian Flow Matching" Miller et al. (2024)      
"FlowLLM: Flow Matching for Material Generation with Large Language Models as Base Distributions" Sriram et al. (2024)   
"Metric Flow Matching for Smooth Interpolations on the Data Manifold" Kapuśniak et al. (2024)   

