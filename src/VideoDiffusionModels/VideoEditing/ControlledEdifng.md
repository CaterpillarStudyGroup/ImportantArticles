
P188    
# 3 Video Editing

## 3.3 Controlled Edifng (depth/pose/point/ControlNet)

> &#x2705; 已有一段视频，通过 guidance 或文本描述，修改视频。    

P189   

![](../../assets/08-189.png) 

P190   
## Depth Control

> &#x2705; RunwayML 主要做的是 style transfer, 强制加入 depth 作为 condition, 因此可移植性非常高。   

P191    
> &#x2705; MIDS 是已有的深度估计模型。   

P192   
## Use MiDaS to offer depth condition

Depth estimating network


Ranftl et al., “Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,” TPAMI 2022.

> &#x2705; 深变信息 Encode 成 latent code, 与 noise conca 到一起。   

P193   
## Gen-1

Framewise depth-guided video editing

 - Inflate Stable Diffusion to a 3D model, finetune on pretrained weights   
 - Insert temporal convolution/attention layers   
 - Finetune to take **per-frame depth as conditions**   

|||
|--|--|
| ![](../../assets/08-193-1.png)  | ![](../../assets/08-193-2.png)  |

> &#x2705; 特点：(1) 不需要训练。 (2) 能保持前后一致性。   

P60   
### Gen-1

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Structure and Content-Guided Video Synthesis with Diffusion Models|Transfer the style of a video using text prompts given a “driving video”，以多种形式在预训练图像扩散模型中融入时序混合层进行扩展|![](../../assets/D3-60.png) |  

P61   
### Gen-1

 - Condition on structure (depth) and content (CLIP) information.   
 - Depth maps are passed with latents as input conditions.   
 - CLIP image embeddings are provided via cross-attention blocks.   
 - During inference, CLIP text embeddings are converted to CLIP image embeddings.    

![](../../assets/D3-61.png)     

> &#x2705; 用 depth estimator 从源视频提取 struct 信息，用 CLIP 从文本中提取 content 信息。   
> &#x2705; depth 和 content 分别用两种形式注入。depth 作为条件，与 lantent concat 到一起。content 以 cross attention 的形式注入。    

P194   
## Pix2Video

Framewise depth-guided video editing

 - Given a sequence of frames, generate a new set of images that reflects an edit.   
 - Editing methods on individual images fail to preserve temporal information.    

Ceylan et al., <u>"Pix2Video: Video Editing using Image Diffusion",</u> arXiv 2023    

> &#x2705; 没有 3D diffusion model，只是用 2D diffusion model 生成多张图像并拼成序列。关键在于保持时序的连续性。    

 - Leverage a pretrained per-frame depth-conditioned Stable Diffusion model to edit frame by frame, to maintain motion consistency between source video and edited video
 - No need for training/finetuning

![](../../assets/08-194.png) 

P195   

### How to ensure temporal consistency?   

#### Obtain initial noise from DDIM inversion   

![](../../assets/08-195.png)  

> &#x2705; (1) 用每一帧的原始图像的 inversion 作为 init noise.   
> &#x2705; (2) 下一帧的生成会引用上一帧的 latent.    
> &#x2705; (3) 生成的中间结果上也会有融合。   


P196   
#### **Self-Attention injection:** 

Inject self-attention features from the previous frame in U-Net for generating the current frame    

![](../../assets/D3-63-1.png)   

- Use the latent of the previous frame as keys and values to guide latent update of the current frame   

![](../../assets/D3-63-2.png)    

![](../../assets/D3-64.png)    

> &#x2705; reconstruction guidance，使生成的 latent code 与上一帧接近。    

> &#x2705; (1) 使用 DDIM inversion 把图像转为 noise．   
> &#x2705; (2) 相邻的 fram 应 inversion 出相似的 noise．    
> &#x2705; 使用 self-attention injection 得到相似的 noise.

P197   
### Result

![](../../assets/08-197.png) 

P198   

![](../../assets/08-198.png) 

P199   
## ControlNet / Multiple Control

P200   
## ControlVideo (Zhang et al. 2023)

提出无需训练的框架，通过结构一致性实现可控文本到视频生成。

ControlNet-like video editing

 - Input structural conditions through **ControlNet**

![](../../assets/08-200.png) 


Zhang et al., “ControlVideo: Training-free Controllable Text-to-Video Generation,” arXiv 2023.    

> &#x2705; 使用预训练的 stable diffusion, 无需额外训练。   
> &#x2705; contrd net 是与 stable diffusion 配对的。   
> &#x2705; contrd net 以深度图或边缘图为条件，并在时间维度上 embed 以此得到的Z。与原始视频有比较好的对应关系，但仍存在 temporal consistency 问题。   


P201  

 - Use pretrained weights for Stable Diffusion & ControlNet, no training/finetuning   
 - Inflate Stable Diffusion and ControlNet along the temporal dimension   
 - Interleaved-frame smoothing during DDIM sampling for bever temporal consistency    

![](../../assets/08-201.png)    

> &#x2705; 解决 temporal consistency 问题，方法：   
> &#x2705; 在每个 timestep，让不同帧成为前后两帧的融合。    
> &#x2753; control net 与 diffusion medel 是什么关系？     


P202   

![](../../assets/08-202.png)

P203   

![](../../assets/08-203.png)  

P207    

> &#x2705; 除了 control net, 还使用光流信息作为引导。   
> &#x2705; Gop：Group of Pictures.    


P208   
## VideoControlNet

Optical flow-guided video editing; I, P, B frames in video compression

![](../../assets/08-208.png)  

Hu et al., “VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet,” arXiv 2023.     

> &#x2705; 内容一致性，适用于 style transfer, 但需要对物体都较大编辑力度时不适用(例如编辑物体形状)。   

P209    

> &#x2705; 也是control net 形式，但用到更多控制条件。   

P210   
## CCEdit

Mulemodal-guided video edieng

![](../../assets/08-210.png)  

Feng et al., “CCEdit: Creative and Controllable Video Editing via Diffusion Models,” arXiv 2023.    

> &#x2705; 使用了更多控制信息，并把它们 combine 到一起。   


P211   
## VideoComposer

Image-, sketch-, motion-, depth-, mask-controlled video editing

**Video Editing based on Various Conditions**

![](../../assets/08-211.png)  

Wang et al., “VideoComposer: Compositional Video Synthesis with Motion Controllability,” arXiv 2023.   

> &#x2705; 每个 condition 进来，都过一个 STC-Encoder, 然后把不同 condition fuse 到一起，输入到 U-Net.    


P212   
## VideoComposer

Image-, sketch-, motion-, depth-, mask-controlled video editing   

• Spako-Temporal Condikon encoder (STC-encoder): a unified input interface for condikons   

![](../../assets/08-212.png)  

Wang et al., “VideoComposer: Compositional Video Synthesis with Motion Controllability,” arXiv 2023.    

P214   
## ControlNet- and Depth-Controlled Video Editing: More Works

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Control-A-Video: Controllable Text-to-Video Generagon with Diffusion Models|通过边缘图或深度图等序列化控制信号生成视频，并提出两种运动自适应噪声初始化策略|![](../../assets/08-214-3.png)|轨迹控制|
||2024|Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models.|轨迹控制|

|||
|--|--|
| ![](../../assets/08-214-1.png)  | **MagicProp** (Yan et al.) <br> “MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation,” arXiv 2023. |
| ![](../../assets/08-214-2.png) | **Make-Your-Video** (Xing et al.) <br> “Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance,” arXiv 2023.   |
| ![](../../assets/08-214-4.png)  | **MagicEdit** (Liew et al.) <br> “MagicEdit: High-Fidelity and Temporally Coherent Video Editing,” arXiv 2023. |
| ![](../../assets/08-214-5.png)  |  **EVE** (Chen et al.) <br> “EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints,” arXiv 2023. |

P215   
## Pose Control

P216  
## DreamPose

Pose- and image-guided video generation

Input: image  \\(\quad \\) Input: pose sequence   \\(\quad \\)  Output: Video   

 
Karras et al., “DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion,” arXiv 2023.   




P218   
## MagicAnimate

Pose- and image-guided video generaeon

**Challenges**

 - Flickering video   
 - Cannot maintain background   
 - Short video animation results   

**Possible Cause**

 - Weak appearance preservation due to lack of temporal modeling    



Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.    

> &#x2705; 把 pose control net 加到核心的 U-Net 生成。   
> &#x2705; 把原始 U-Net fix, copy- 分可以 train 的 U-Net.    
> &#x2705; 输入：reference image, 两个 U-Net 在部分 layer 进行结合达到前景 appearance 和背景 appeorance 的 Encode 推断时输入多个 Sequence, 可以生成 long video.   


P219   
## MagicAnimate

Pose- and image-guided video generation   


![](../../assets/08-219.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.    

P220   
## MagicAnimate

Pose- and image-guided video generation

![](../../assets/08-220.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.



P223   
## MagicAnimate

Pose-guided video generation

![](../../assets/08-223.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.    

P224   
## Video Editing Under Pose Guidance: More Works

|||
|--|--|
| ![](../../assets/08-224-1.png)  | **Dancing Avatar** (Qin et al.)<br> Pose-guided video editing <br> “Dancing avatar: Pose and text-guided human motion videos synthesis with image diffusion model,” arXiv 2023. |
| ![](../../assets/08-224-3.png)  | **DisCo** (Wang et al.) <br> Pose-guided video editing <br> “Disco: Disentangled control for referring human dance generation in real world,” arXiv 2023.  |

P225   
## Point-Control

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|98|2023|VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence|

P226   
  

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/