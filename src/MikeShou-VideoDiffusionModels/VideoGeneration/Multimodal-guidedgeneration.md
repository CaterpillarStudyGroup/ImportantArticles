
P134  
# 2 Video Generation

## 2.7 Multimodal-guided generation

P135  

![](../../assets/08-135.png) 

> &#x2705; 需要训练的模型，且针对一个模型进行训练。   


P136  
## MCDiff  

Motion-guided video generation

![](../../assets/08-136.png) 


Chen et al., “Motion-Conditioned Diffusion Model for Controllable Video Synthesis,” arXiv 2023.    

P137  
## MCDiff

Motion-guided video generation

 - Two-stage autoregressive generation

![](../../assets/08-137.png) 

Chen et al., “Motion-Conditioned Diffusion Model for Controllable Video Synthesis,” arXiv 2023.    

> &#x2705; 基本泛式：输入：一段视频，一个文生图模型，一个文本提示词。输出：基于定制化的文生图得到文生视频。   
> &#x2705; 不在大规模上训练，只在一个视频上训练，只需十分钟。   


P139   
## The Power of Sound (TPoS)

Sound- and text-guided video generation

 - Input/output: a text prompt + an audio segment → a video

![](../../assets/08-139.png) 

Jeong et al., “The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion,” ICCV 2023.    

> &#x2705; 用文生图模型生成 appearance, dynamics 来自于 reference video.    


P140
## Sound-Guided Video Generation: More Works

|||
|--|--|
| ![](../../assets/08-140-1.png)  | **AADiff** (Lee et al.) <br> “AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion,” CVPRW 2023. |
| ![](../../assets/08-140-2.png) | **Generative Disco** (Liu et al.)<br> “Generative Disco: Text-to-Video Generation for Music Visualization,” arXiv 2023. |

Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.   

> &#x2705; 一些观察：文生图模型能生成动词，但动作没有一致性。可以通过引入 temperal attention 保证一致性。    


P141  
## Image-Guided Video Generation: More Works

|||
|--|--|
| ![](../../assets/08-141-1.png)  | Generative Image Dynamics (Li et al.) <br> “Generative Image Dynamics,” arXiv 2023. |
|  ![](../../assets/08-141-2.png)  | **LaMD** (Hu et al.) <br> “LaMD: Latent Motion Diffusion for Video Generation,” arXiv 2023. |
|  ![](../../assets/08-141-3.png)  | **LFDM** (Ni et al.) <br> “Conditional Image-to-Video Generation with Latent Flow Diffusion Models,” CVPR 2023.  |


Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.   


> &#x2705; 当前帧只与上帧和前一帧做 attention，大大减少计算量。  
> &#x2705; 在所有帧上做 attention 开销比较大。   
> &#x2705; 解决方法：前一帧与第一帧。   
> &#x2753; 怎么保证生成动作与原视频动作的一致性呢?    


P142  
## Cinematic Mindscapes   

Brain activity-guided video generation

 - Task: human vision reconstruction via fMRI signal-guided video generation   

![](../../assets/08-142.png) 

Chen et al., “Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity,” arXiv 2023.     

> &#x2705; 对要编辑的视频，先 DDIM Inversion，得到 inverfed noise，这是保留了原视频 pattern 的 noise.   
> &#x2705; 用这个 noise 作为 init noise，还原出的视频跟原视频有比较好的结构化保留。   
> &#x2705; 解法方法


P144  
## Multimodal-Guided Video Generation: More Works


|||
|--|--|
| ![](../../assets/08-144-1.png)  | **MovieFactory** (Zhu et al.) <br> “MovieFactory: Automatic Movie Creation from Text using Large Generative Models for Language and Images,” arXiv 2023. |
| ![](../../assets/08-144-2.png) | **CoDi** (Tang et al.) <br> “Any-to-Any Generation via Composable Diffusion,” NeurIPS 2023. |
| ![](../../assets/08-144-3.png)  | **MM-Diffusion** (Ruan et al.) <br> “MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation,” CVPR 2023. |
|  ![](../../assets/08-144-4.png) | **NExT-GPT** (Wu et al.) <br> “NExT-GPT: Any-to-Any Multimodal LLM,” arXiv 2023.  |
|  |  |


Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.

> &#x2705; 在物体改变比较大的情况下，diffusion 比其它生成方法效果更好。   


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/