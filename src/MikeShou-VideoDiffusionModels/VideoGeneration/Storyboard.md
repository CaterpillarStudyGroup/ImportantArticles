


P108  
# 2 Video Generation

## 2.5 Storyboard

> &#x2705; Control Net，把文本转为 Pixel 图片。  

 
P109  
![](../../assets/08-109.png) 

> &#x2705; 用 GPT-4 In-context learning 机制生成结构化文本     


P110 
## What is a storyboard?


Human can imagine what does the scene look like “roughly”

“Two men stand in the airport waiting room and stare at the 
airplane thru window”   

What is in your mind now?

Storyboard image from deviantart.com.

> &#x2705; 难点：保持内容的一致性。   


P111
## What is a storyboard?

A concept in film production

![](../../assets/08-111.png) 

 - Rough sketches/drawings with notes    
 - Example: Inception by Christopher Nola   

Storyboard image from deviantart.com.    


P112 
## What is a storyboard?

A concept in film production

 - How to generate such a storyboard?    
 - As humans, over the years, we have acquired such “visual prior” about object location, object shape, relation, etc.   

 - Can LLM model such visual prio？    

Storyboard image from deviantart.com.   


> &#x2705; 没有训练 GPT／LLM，而是使用文本来引导，但是生成结果不合理。   
> &#x2705; GPT 缺少一些视觉上的 commen sense 主要是缺少相关数据集。  
> &#x2705; 因此这里提供了一个数据集。   



P113   
## VisorGPT

Can we model such visual prior with LLM

![](../../assets/08-113.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.

P114   
## VisorGPT

Prompt design

![](../../assets/08-114-1.png) 

![](../../assets/08-114-2.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.   


P116    
> &#x2705; 两层 diffusion    
> &#x2705; 通过 recursive 的插帧生成非常长的视频。   


P118   
## VisorGPT

Modeling Visual Prior via Generative Pre-Training

![](../../assets/08-118.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.    

> &#x2705; 递归的 Local Diffusion    


P119  
## VisorGPT

Sample from the LLM which has learned visual prior

![](../../assets/08-119.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.    

> &#x2705; Global：文生图  \\(\quad\\)  Local：图序列补全。   
> &#x2705; Global 和 Local 使用相似的模型，训练方法不同，主要是 MASK 的区别。   


P120   
## VisorGPT

Sample from the LLM which has learned visual prior

![](../../assets/08-120.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.  

P121   
## VideoDirectorGPT

Use storyboard as condition to generate video

![](../../assets/08-121.png) 

Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,” arXiv 2023.   


P122  
## VideoDirectorGPT

Use storyboard as condition to generate video

![](../../assets/08-122.png) 

Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,” arXiv 2023.    

P124   
## Long-form Video Prior

GPT can be trained to learn better long-form video prior (e.g., object position, relative size, human interaction)

**A new dataset - Storyboard20K**

![](../../assets/08-124.png) 

Xie et al., “Learning Long-form Video Prior via Generative Pre-Training,” to be released in 2024.    
<https://github.com/showlab/Long-form-Video-Prior>   


P125  
## Storyboard: More Works
  
|||
|--|--|
|  ![](../../assets/08-125-1.png) | **Dysen-VDM** (Fei et al.)<br>Storyboard through scene graphs<br>“Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models,” arXiv 2023. |
| ![](../../assets/08-125-2.png)  | **DirectT2V** (Hong et al.) <br> Storyboard through bounding boxes <br> “Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation,” arXiv 2023. |
|  ![](../../assets/08-125-3.png)  | **Free-Bloom** (Huang et al.)<br>Storyboard through detailed text prompts<br> “Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator,” NeurIPS 2023. |
|  ![](../../assets/08-125-4.png) | **LLM-Grounded Video Diffusion Models** (Lian et al.) <br> Storyboard through foreground bounding boxes <br> “LLM-grounded Video Diffusion Models,” arXiv 2023. |


> &#x2705; (1) 画运动轨迹  (2) 光流 (3) 做为 condition，可以细粒度地控制运动轨迹。   
> &#x2705; (1) 画出的轨迹生成 derse 光流。    
> &#x2705; (2) 用光流作为 Condition 生成。   


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/