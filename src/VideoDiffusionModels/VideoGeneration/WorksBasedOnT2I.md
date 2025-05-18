P75   
# Text-2-Video


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

##  More Works

|||
|--|--|
| ![](../../assets/08-107-1.png)  | **MagicVideo** (Zhou et al.) <br> Insert causal attention to Stable Diffusion for better temporal coherence <br> “MagicVideo: Efficient Video Generation With Latent Diffusion Models,” arXiv 2022.  |
|  ![](../../assets/08-107-2.png)  | **Simple Diffusion Adapter** (Xing et al.) <br> Insert lightweight adapters to T2I models, shift latents, and finetune adapters on videos <br>“SimDA: Simple Diffusion Adapter for Efficient Video Generation,” arXiv 2023. |
| ![](../../assets/08-107-3.png) | **Dual-Stream Diffusion Net** (Liu et al.) <br> Leverage multiple T2I networks for T2V <br> “Dual-Stream Diffusion Net for Text-to-Video Generation,” arXiv 2023. |
||MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation,2024|


> &#x2705; 用纯文本的形式把图片描述出来。   
> &#x2705; 方法：准备好 pair data，对 GPT 做 fine-tune.    
> &#x2705; 用结构化的中间表示生成图片。   
> &#x2705; 先用 GPT 进行文本补全。   


# Image-2-Video

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|51|2023|Motion-Conditioned Diffusion Model for Controllable Video Synthesis|&#x2705; 用户提供的稀疏运动轨迹 -> dense光流<br>&#x2705; dense光流（condition） + Image -> 视频|Two-stage,  自回归生成|[link](https://caterpillarstudygroup.github.io/ReadPapers/51.html)|
|44|2024|Motion-I2V: Consistent and Controllable Image-to-Video Generation with Explicit Motion Modeling|&#x2705; 用户提供的控制信号（condition）+ Image -> dense光流<br>&#x2705; dense光流（condition） + Image -> 视频|Two-stage，轨迹控制|[link](https://caterpillarstudygroup.github.io/ReadPapers/44.html)|
||2024|Physmotion: Physicsgrounded dynamics from a single image.|轨迹控制|
| |2023| **LFDM** (Ni et al.) <br> “Conditional Image-to-Video Generation with Latent Flow Diffusion Models,”|&#x2705;视频->光流 + Mask<br>&#x2705; 光流+Mask+图像 ->视频|![](../../assets/08-141-3.png)  ||
||2024|Generative Image Dynamics (Li et al.) <br> “Generative Image Dynamics,”|图像（无condition） -> SV <br>&#x2705; SV + 力 -> 光流 <br>&#x2705; 光流 + Image -> 视频| ![](../../assets/08-141-2.png)  ||
||2023|LaMD: Latent Motion Diffusion for Video Generation|视频 -> 图像特征 + 运动特征<br>&#x2705; 运动特征+图像特征->视频|![](../../assets/08-141-2.png)  |
||2023|Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models| ![](../../assets/08-95-3.png) | **PYoCo** (Ge et al.)<br> Generate video frames starting from similar noise patterns|
||2023|Animate-a-story: Storytelling with retrieval-augmented video generation|深度控制|
1.    2

## More Works 闭源
|||
|--|--|
| ![](../../assets/08-95-1.png)  | **Latent Shift** (An et al.)<br>Shift latent features for better temporal coherence <br> “Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-2.png) | **Video Factory** (Wang et al.)<br> Modify attention mechanism for better temporal coherence <br> “VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-4.png)  | **VideoFusion** (Lorem et al.)<br> Decompose noise into shared “base” and individual “residuals”<br>“VideoFusion: ecomposed Diffusion Models for High-Quality Video Generation,” CVPR 2023. |

> &#x2705; Framwork (1) 在原模型中加入 temporal layers (2) fix 原模型，训练新的 layers (3) 把 lager 插入到目标 T2 I 模型中。   

# Sound2Video

## The Power of Sound (TPoS)

Sound- and text-guided video generation

 - Input/output: a text prompt + an audio segment → a video

![](../../assets/08-139.png) 

Jeong et al., “The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion,” ICCV 2023.    

## More Works

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|**AADiff**: Audio-Aligned Video Synthesis with Text-to-Image Diffusion||![](../../assets/08-140-1.png)  | 
||2023|**Generative Disco** (Liu et al.)<br> “Generative Disco: Text-to-Video Generation for Music Visualization,| | ![](../../assets/08-140-2.png) | 

# Bain Activity 2 Video   

> &#x2705; 大脑信号控制生成。   

Brain activity-guided video generation

 - Task: human vision reconstruction via fMRI signal-guided video generation   

![](../../assets/08-142.png) 

Chen et al., “Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity,” arXiv 2023.   

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/
