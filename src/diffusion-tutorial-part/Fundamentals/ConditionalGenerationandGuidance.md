P66   
# Conditional Generation and Guidance    

P67   
> &#x2705; 通常需要的是特定的生成，而不是随意的生成。因此需要通过control引入特定的需求。  

以下是文生图的例子：

![](../../assets/D1-67.png) 

<u>Ramesh et al., “Hierarchical Text-Conditional Image Generation with CLIP Latents”, arXiv 2022.</u>    
<u>Saharia et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding”, arXiv 2022.</u>    

P68   
## Conditioning and Guidance Techniques

Explicit Conditions    
Classifier Guidance    
Classifier-free Guidance    

P69   

# Explicit Conditions    

P70   
Conditional sampling can be considered as training \\(p(\mathbf{x} |\mathbf{y} )\\) where \\(\mathbf{y}\\) is the input conditioning (e.g., text) and \\(\mathbf{x}\\) is generated output (e.g., image)    

Train the score model for \\(\mathbf{x}\\) conditioned on \\(\mathbf{y}\\) using:    

$$
\mathbb{E} _ {(\mathbf{x,y} )\sim P\mathrm{data} (\mathbf{x,y} )}\mathbb{E} _ {\epsilon \sim \mathcal{N}(\mathbf{0,I} ) }\mathbb{E} _{t\sim u[0,T]}||\epsilon _ \theta (\mathbf{x} _ t,t;\mathbf{y} )- \epsilon ||^2_2 
$$

The conditional score is simply a U-Net with \\(\mathbf{x}_t\\) and \\(\mathbf{y}\\) together in the input.    

![](../../assets/D1-70.png) 

> &#x2705; 需要 \\((x，y)\\) 的 pair data.            

P71   

# Classifier Guidance    

P72   
## Bayes’ Rule in Action

![](../../assets/D1-72.png) 

> &#x2705; \\(p(y)\\) 与 \\(\mathbf{x} _ t\\) 无关，因此可以去掉。     

## 训练方法

> &#x2705; 第一步：需要一个训好的p(x)的 diffusion model 。  
> &#x2705; 第二步：训练一个分类网络，输入xt能够正确地预测控制条件（y不一定是离散的类别）。  
> &#x2705; 第三步：取第二步的梯度，用一定的权重\\(w \\)结合到第一步的forward过程中。\\(w \\)决定分类器的影响力。   

> &#x2705; 只需要部分pair data和大量的非pair data。但需要单独训练一个分类器。  

## 相关论文

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2021|Controllable and Compositional Generation with Latent-Space Energy-Based Models|
||2021|Diffusion models beat GANs on image synthesis|

# Classifier-free Guidance    

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2021|Classifier-Free Diffusion Guidance|||[link](https://caterpillarstudygroup.github.io/ReadPapers/6.html)|     

