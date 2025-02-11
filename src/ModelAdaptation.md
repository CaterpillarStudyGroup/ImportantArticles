P111  
# Model Adaptation  

P112    
## You’ve trained a model. What next?

![](assets/P112图.png)    

P113    
Faster Sampling   

P114    
## Faster sampling by straightening the flow   

![](assets/P114图.png)    

$$
ℒ(θ) = \mathbb{E}_{t,(X_0,X_1)∼π_{0,1}^0}||u^θ_t (X_t) − (X_1 − X_0)||^2
$$

Rectified Flow refits using the **pre-trained (noise, data) coupling**.      
**Leads to straight flows**.     

“Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow” Liu et al. (2022)      
“InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation” Liu et al. (2022)    

