P75   
# Works Based on T2I Base Models

## Show-1

Better text-video alignment? Generation in both pixel- and latent-domain

![](../../assets/08-75.png) 

> &#x2705; Stable Diffusion Model存在的问题：当文本变复杂时，文本和内容的 align 不好。  
> &#x2705; show-1 在 alignment 上做了改进。   

Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.    

P76   
### **Motivation**

#### pixel VS latent: 一致性
 - Pixel-based VDM achieves better text-video alignment than latent-based VDM   

|||
|--|--|
| ![](../../assets/08-76-1.png) | ![](../../assets/08-76-2.png) |

> &#x2705; 实验发现：pixel spase 比 latent space 更擅长 align ment.   
> &#x2705; 原因：在 latent space，文本对 pixel 的控制比较差。   

P77   
#### pixel VS latent: memory

 - Pixel-based VDM achieves better text-video alignment than latent-based VDM   
 - Pixel-based VDM takes much larger memory than latent-based VDM    

![](../../assets/08-77.png)  


P78   
### 本文方法 

 - Use Pixel-based VDM in low-res stage   
 - Use latent-based VDM in high-res stage   

![](../../assets/08-78.png) 


P79   
### Result

<https://github.com/showlab/Show-1>

 - Better text-video alignment   
 - Can synthesize large motion   
 - Memory-efficient   

Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.   

P98 
## AnimateDiff  

Guo et al., “AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning,” arXiv 2023.     

### T2I -> T2V

Transform domain-specific T2I models to T2V models

 - Domain-specific (personalized) models are widely available for image   
    - Domain-specific finetuning methodologies: LoRA, DreamBooth…   
    - Communities: Hugging Face, CivitAI…   
 - **Task: turn these image models into T2V models, without specific finetuning**   


> &#x2705; (1) 用同一个 patten 生成 noise，得到的 image 可能更有一致性。   
> &#x2705; (2) 中间帧的特征保持一致。    


P99  
### **Methodology**

 - Train a motion modeling module (some temporal layers) together with frozen base T2I model   
 - Plug it into a domain-specific T2I model during inference   

![](../../assets/08-99.png) 

> &#x2705; 优势：可以即插即用到各种用户定制化的模型中。   
> &#x2705; 在 noise 上对内容进行编辑，即定义第一帧的 noise，以及后面帧的 noise 运动趋势。   


P100 
![](../../assets/08-100.png)   

### Training

 - Train on WebVid-10M, resized at 256x256 (experiments show can generalize to higher res.)   

> &#x2705; 在低分辨率数据上训练，但结果可以泛化到高分辨率。   

> &#x2705; 保证中间帧尽量相似。   

P101   
> &#x2705; 扣出背景并 smooth.    

P102  
## Text2Video-Zero   

Use Stable Diffusion to generate videos without any finetuning

> &#x2705; 完全没有经过训练，使用 T2I Base Model(stable diffusion Model) 生成视频。  

**Motivation: How to use Stable Diffusion for video generation without finetuning?**  

 - Start from noises of similar pattern   
 - Make intermediate features of different frames to be similar   


Khachatryan et al., “Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators,” arXiv 2023.    

P103   

### Step 1
 - Start from noises of similar pattern: given the first frame’s noise, define a global scene motion, used to translate the first frame’s noise to generate similar initial noise for other frames   

![](../../assets/08-103.png) 

P104   
### Step2
 - Make intermediate features of different frames to be similar: always use K and V from the first frame in self-attention   

![](../../assets/08-104.png)  


> &#x2705; 生成电影级别的视频，而不是几秒钟的视频。   


P105   

### Step3

 - Optional background smoothing: regenerate the background, average with the first frame

![](../../assets/08-105.png) 

P106   
> &#x2705; 文本 → 结构化的中间脚本 → 视频   


## Multi-Modal Guided Generation

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|51|2023|Motion-Conditioned Diffusion Model for Controllable Video Synthesis|<br> &#x2705; (1) 把用户提供的稀疏运动轨迹转为dense光流<br> &#x2705; (2) 用光流作为 Condition 生成视频。|Two-stage,  自回归生成|[link](https://caterpillarstudygroup.github.io/ReadPapers/51.html)|
|44|2024|Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling|<br> &#x2705; &#x2705; (1) 用光流作为 Condition 生成视频。<br> (2) 把用户提供的控制信号转为dense光流，从而控制图像生成。|Two-stage|[link](https://caterpillarstudygroup.github.io/ReadPapers/44.html)|


##  More Works

|||
|--|--|
| ![](../../assets/08-107-1.png)  | **MagicVideo** (Zhou et al.) <br> Insert causal attention to Stable Diffusion for better temporal coherence <br> “MagicVideo: Efficient Video Generation With Latent Diffusion Models,” arXiv 2022.  |
|  ![](../../assets/08-107-2.png)  | **Simple Diffusion Adapter** (Xing et al.) <br> Insert lightweight adapters to T2I models, shift latents, and finetune adapters on videos <br>“SimDA: Simple Diffusion Adapter for Efficient Video Generation,” arXiv 2023. |
| ![](../../assets/08-107-3.png) | **Dual-Stream Diffusion Net** (Liu et al.) <br> Leverage multiple T2I networks for T2V <br> “Dual-Stream Diffusion Net for Text-to-Video Generation,” arXiv 2023. |

> &#x2705; 用纯文本的形式把图片描述出来。   
> &#x2705; 方法：准备好 pair data，对 GPT 做 fine-tune.    
> &#x2705; 用结构化的中间表示生成图片。   
> &#x2705; 先用 GPT 进行文本补全。   

## More Works 闭源
|||
|--|--|
| ![](../../assets/08-95-1.png)  | **Latent Shift** (An et al.)<br>Shift latent features for better temporal coherence <br> “Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-2.png) | **Video Factory** (Wang et al.)<br> Modify attention mechanism for better temporal coherence <br> “VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-3.png) | **PYoCo** (Ge et al.)<br> Generate video frames starting from similar noise patterns <br> “Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models,” ICCV 2023.  |
| ![](../../assets/08-95-4.png)  | **VideoFusion** (Lorem et al.)<br> Decompose noise into shared “base” and individual “residuals”<br>“VideoFusion: ecomposed Diffusion Models for High-Quality Video Generation,” CVPR 2023. |

> &#x2705; Framwork (1) 在原模型中加入 temporal layers (2) fix 原模型，训练新的 layers (3) 把 lager 插入到目标 T2 I 模型中。   

# Works Based on T2V Base Models

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/
