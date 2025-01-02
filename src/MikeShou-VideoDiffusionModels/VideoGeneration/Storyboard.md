P108 
# 2.5 Storyboard
 
P109  
![](../../assets/08-109.png) 

P110 
## What is a storyboard?


Human can imagine what does the scene look like “roughly”

“Two men stand in the airport waiting room and stare at the 
airplane thru window”   

What is in your mind now?

> &#x2705; 难点：保持内容的一致性。   


P111
### What is a storyboard?

A concept in film production

![](../../assets/08-111.png) 

 - Rough sketches/drawings with notes    
 - Example: Inception by Christopher Nola   

Storyboard image from deviantart.com.    


P112 
### What is a storyboard?

A concept in film production

 - How to generate such a storyboard?    
 - As humans, over the years, we have acquired such “visual prior” about object location, object shape, relation, etc.   

 - Can LLM model such visual prio？      



P113   
## VisorGPT

Can we model such visual prior with LLM

![](../../assets/08-113.png) 

P114   

### Prompt design

![](../../assets/08-114-1.png) 

![](../../assets/08-114-2.png)  

P118  

### Modeling Visual Prior via Generative Pre-Training

![](../../assets/08-118.png)  


P119  

### Sample from the LLM which has learned visual prior

![](../../assets/08-119.png) 

P120   

![](../../assets/08-120.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.  

P121   
## VideoDirectorGPT

Use storyboard as condition to generate video

![](../../assets/08-121.png) 


> &#x2705; Control Net，把文本转为 Pixel 图片。

P122  

![](../../assets/08-122.png) 

Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,” arXiv 2023.    

P124   
## Long-form Video Prior

GPT can be trained to learn better long-form video prior (e.g., object position, relative size, human interaction)

> &#x2705; 用 GPT-4 In-context learning 机制生成结构化文本     

**A new dataset - Storyboard20K**

![](../../assets/08-124.png) 

Xie et al., “Learning Long-form Video Prior via Generative Pre-Training,” to be released in 2024.    
<https://github.com/showlab/Long-form-Video-Prior>   

> &#x2705; 没有训练 GPT／LLM，而是使用文本来引导，但是生成结果不合理。   
> &#x2705; GPT 缺少一些视觉上的 commen sense 主要是缺少相关数据集。  
> &#x2705; 因此这里提供了一个数据集。   

P125  
## Storyboard: More Works
  
|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|41|2024|STORYDIFFUSION: CONSISTENT SELF-ATTENTION FOR LONG-RANGE IMAGE AND VIDEO GENERATION|先生成一致的关键帧，再插帧成中间图像||[link](https://caterpillarstudygroup.github.io/ReadPapers/41.html)|

|||
|--|--|
|  ![](../../assets/08-125-1.png) | **Dysen-VDM** (Fei et al.)<br>Storyboard through scene graphs<br>“Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models,” arXiv 2023. |
| ![](../../assets/08-125-2.png)  | **DirectT2V** (Hong et al.) <br> Storyboard through bounding boxes <br> “Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation,” arXiv 2023. |
|  ![](../../assets/08-125-3.png)  | **Free-Bloom** (Huang et al.)<br>Storyboard through detailed text prompts<br> “Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator,” NeurIPS 2023. |
|  ![](../../assets/08-125-4.png) | **LLM-Grounded Video Diffusion Models** (Lian et al.) <br> Storyboard through foreground bounding boxes <br> “LLM-grounded Video Diffusion Models,” arXiv 2023. |


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/