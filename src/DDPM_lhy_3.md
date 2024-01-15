
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






![](./assets/lhy3-17.png)  