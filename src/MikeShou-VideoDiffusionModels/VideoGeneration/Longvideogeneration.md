P126   
# 2.6 Long video generation
P127  

![](../../assets/08-127.png) 

> &#x2705; 声音(非语音)引导的生成。  


P128  
## NUWA-XL  

Recursive interpolations for generating very long videos


> &#x2705; 两层 diffusion    
> &#x2705; 通过 recursive 的插帧生成非常长的视频。   

### Method Proposed

 - A “diffusion over diffusion” architecture for very long video generation

### **Key Idea**

 - Key idea: coarse-to-fine hierarchical generation

### **Other Highlights**

 - Trained on very long videos (3376 frames)
 - Enables parallel inference
 - Built FlintstonesHD: a new dataset for long video generation, contains 166 episodes with an average of 38000 frames of 1440 × 1080 resolution

> Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,” arXiv 2023.   

### **Generation Pipeline**

 - Storyboard through multiple text prompts   

![](../../assets/08-129.png)    


P130  

 - Global diffusion model: L text prompts → L keyframes
 - Local diffusion model: 2 text prompts + 2 keyframes → L keyframes   

![](../../assets/08-130.png) 

> &#x2705; Global：文生图  \\(\quad\\)  Local：图序列补全。   
> &#x2705; Global 和 Local 使用相似的模型，训练方法不同，主要是 MASK 的区别。   

> &#x2705; 大脑信号控制生成。   


P131  

### **Mask Temporal Diffusion (MTD)**  

 - A basic diffusion model for global & local diffusion models

![](../../assets/08-131.png)  

> &#x2705; 递归的 Local Diffusion    

P133
## Long Video Generation: More Works

|||
|--|--|
| ![](../../assets/08-133-1.png)  | **Latent Video Diffusion Models for High-Fidelity Long Video Generation** (He et al.) <br> Generate long videos via autoregressive generation & interpolation <br> “Latent Video Diffusion Models for High-Fidelity Long Video Generation,” arXiv 2022.|
|  ![](../../assets/08-133-2.png) | **VidRD** (Gu et al.) <br> Autoregressive long video generation <br> “Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation,” arXiv 2023. |
|  ![](../../assets/08-133-3.png) | **VideoGen** (Li et al.) <br> Cascaded pipeline for long video generation <br> “VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation,” arXiv 2023.|


> &#x2705; 已有一段视频，通过 guidance 或文本描述，修改视频。    


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/
