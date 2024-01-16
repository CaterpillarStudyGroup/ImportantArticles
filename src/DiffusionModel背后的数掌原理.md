
P2   
## 基本概念 

![](./assets/lhy3-2.png) 


P3   
## VAE vs. Diffusion Model 

![](./assets/lhy3-2.png) 

P5   
## <u>Training</u> 

![](./assets/lhy3-5.png) 



P6

![](./assets/lhy3-6.png) 

P7   

|||
|--|--|
| 想像中… | ![](./assets/lhy3-7-1.png)  |
| 实际上…  | ![](./assets/lhy3-7-2.png)  |

P8   
## <u> Inference </u> 

![](./assets/lhy3-8-1.png) 


![](./assets/lhy3-8-2.png) 


P10   
## 影像生成模型本质上的共同目标

![](./assets/lhy3-10.png) 

P11   
## Maximum Likelihood Estimation

![](./assets/lhy3-11.png) 

Sample {\\(x^1,x^2,\cdots ,x^m\\)} from \\(P_{data}(x)\\)    

We can compute \\(P_\theta (x^i)\\)    

\begin{align*} \theta ^\ast =\text{arg } \max_{\theta } \prod_{i=1}^{m} P_\theta (x^i) \end{align*}


P11  

![](./assets/lhy3-12.png) 


Maximum Likelihood = Minimize KL Divergence    


P13   
## VAE: Compute \\(𝑃_\theta(x)\\)   

|||
|--|--|
| ![](./assets/lhy3-13-1.png) | ![](./assets/lhy3-13-2.png) |
| ![](./assets/lhy3-13-3.png) | ![](./assets/lhy3-13-4.png) |



P14   
## VAE: Lower bound of \\(log P(x)\\)  

![](./assets/lhy3-14.png)


P15   
## DDPM: Compute \\(𝑃_\theta(x)\\)   

![](./assets/lhy3-15.png)  

\begin{align*} P\theta (x_0)=\int\limits _{x_1:x_T}^{} P(x_T)P_\theta (x_{T-1}|x_T)\dots P_\theta (x_{t-1}|x_t) \dots P_\theta(x_0|x_1)dx_1:x_T  \end{align*}



P16   
## DDPM: Lower bound of \\(log P(x)\\)  

![](./assets/lhy3-16-1.png)  

![](./assets/lhy3-16-2.png)  



P17   
![](./assets/lhy3-17.png)  



P18   
![](./assets/lhy3-18.png)  


P19   
![](./assets/lhy3-19.png)  



P20   
![](./assets/lhy3-20.png)  



P21   
![](./assets/lhy3-21.png)  



P22   
## DDPM: Lower bound of \\(log P(x)\\)  

\begin{align*} E_{q(x_1|x_0)}[log P(x_0|x_1)]-KL(q(x_T|x_0)||P(x_T))\\\\
-\sum_{t=2}^{T}E_{q(x_t|x_0)}[KL(q(x_{t-1}|x_t,x_0)||P(x_{t-1}|x_t))]   \end{align*}


P23  

![](./assets/lhy3-23-1.png)  


![](./assets/lhy3-23-2.png)  


P24   
![](./assets/lhy3-24.png)  


P25   
![](./assets/lhy3-25.png)  

<https://arxiv.org/pdf/2208.11970.pdf>


P26   
![](./assets/lhy3-26.png)  


P27   
![](./assets/lhy3-27-1.png)  

How to minimize KL divergence?    

![](./assets/lhy3-27-2.png)  


![](./assets/lhy3-27-3.png)  



P28   
![](./assets/lhy3-28-1.png)  

![](./assets/lhy3-28-2.png)  


P31   
![](./assets/lhy3-31.png)  


P32   
![](./assets/lhy3-32-1.png)  

![](./assets/lhy3-32-2.png)  

P33   
![](./assets/lhy3-33.png)  

为什么不直接取 Mean？


P34   
## 免责声明：以下只是猜测    

P35   
## 为什么生成文句时需要 Sample？


P36  
## 为什么生成文句时需要 Sample？

 - The Curious Case of Neural Text Degeneration   

<https://arxiv.org/abs/1904.09751>  

![](./assets/lhy3-36.png)  


P37   
![](./assets/lhy3-37.png)  

P39   
## Diffusion Model 是一种 Autoregressive 


![](./assets/lhy3-39.png)  



P43   
## Diffusion Model for Text

 - Difficulty:    
 - Solution: Noise on latent space    

![](./assets/lhy3-43.png)  

<https://arxiv.org/abs/2205.14217>


P44   

![](./assets/lhy3-44.png)  

<https://arxiv.org/abs/2210.08933>


P45   
## Diffusion Model for Text

 - Solution: Don’t add Gaussian noise 

<https://arxiv.org/abs/2210.16886>

Diffusion via Edit￾based Reconstruction (DiffusER)

![](./assets/lhy3-45-1.png)  

![](./assets/lhy3-45-2.png)  


<https://arxiv.org/abs/2107.03006>


P48   
## Mask-Predict 

<https://aclanthology.org/D19-1633/>


![](./assets/lhy3-48.png)  


P49   
## Mask-Predict 

<https://arxiv.org/abs/2202.04200>

<https://arxiv.org/abs/2301.00704>

![](./assets/lhy3-49.png)  

P50   
![](./assets/lhy3-50-1.png)  

![](./assets/lhy3-50-2.png)  

P51   

Scheduled Parallel Decoding with MaskGIT

Sequential Decoding with Autoregressive Transformers


