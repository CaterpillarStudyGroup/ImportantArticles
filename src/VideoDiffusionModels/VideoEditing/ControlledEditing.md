# 3.3 Controlled Edifng (depth/pose/point/ControlNet)

> &#x2705; 已有一段视频，通过 guidance 或文本描述，修改视频。    

P189   

![](../../assets/08-189.png) 

P190   
## Depth Control

Depth estimating network

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2022|Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer| &#x2705; 深变信息 Encode 成 latent code, 与 noise concat 到一起。 |  
|122|2023|Structure and Content-Guided Video Synthesis with Diffusion Models|Transfer the style of a video using text prompts given a “driving video”，以多种形式在预训练图像扩散模型中融入时序混合层进行扩展|![](../../assets/D3-60.png) | Gen-1, Framewise, depth-guided|
|123|2023|Pix2Video: Video Editing using Image Diffusion|Framewise depth-guided video editing|   

P199   
# ControlNet / Multiple Control

也是control net 形式，但用到更多控制条件。   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|124|2023|ControlVideo: Training-free Controllable Text-to-Video Generation|
||2023|VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet|  Optical flow-guided video editing; I, P, B frames in video compression <br> &#x2705; 内容一致性，适用于 style transfer, 但需要对物体有较大编辑力度时不适用(例如编辑物体形状)。 | ![](../../assets/08-208.png)  |
||2023|CCEdit: Creative and Controllable Video Editing via Diffusion Models||![](../../assets/08-210.png)  |
||2023|VideoComposer: Compositional Video Synthesis with Motion Controllability|Image-, sketch-, motion-, depth-, mask-controlled video editing<br> &#x2705; 每个 condition 进来，都过一个 STC-Encoder, 然后把不同 condition fuse 到一起，输入到 U-Net. <br> Spako-Temporal Condikon encoder (STC-encoder): a unified input interface for condikons | ![](../../assets/08-211.png)  <br>![](../../assets/08-212.png) |
||2023|Control-A-Video: Controllable Text-to-Video Generagon with Diffusion Models|通过边缘图或深度图等序列化控制信号生成视频，并提出两种运动自适应噪声初始化策略||![](../../assets/08-214-3.png)|轨迹控制|
||2024|Vmc: Video motion customization using temporal attention adaption for text-to-video diffusion models.|轨迹控制|
||2023|MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation||![](../../assets/08-214-1.png)|
||2023|Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance||![](../../assets/08-214-2.png)|
||2023|MagicEdit: High-Fidelity and Temporally Coherent Video Editing||![](../../assets/08-214-4.png)  |
||2023|EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints|| ![](../../assets/08-214-5.png) 


P225   
# Point-Control

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|98|2023|VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence|

P226   
  

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/