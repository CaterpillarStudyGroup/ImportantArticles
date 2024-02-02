

P89   
# 2 Video Generation

## 2.3 Other closed-source works


P90   

![](../../assets/08-90.png) 


P91   
## GenTron

Transformer-based diffusion for text-to-video generation

 - Transformer-based architecture extended from DiT (class-conditioned transformer-based LDM)   
 - Train T2I \\(\to \\)  insert temporal self-attn \\(\to \\) joint image-video finetuning (motion-free guidance)    

![](../../assets/08-91.png) 

Chen et al., “GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation,” arXiv 2023.    

P93   
## W.A.L.T.

Transformer-based diffusion for text-to-video generation  

 - Transformer-based denoising diffusion backbone   
 - Joint image-video training via unified image/video latent space (created by a joint 3D encoder with causal 3D conv layers, allowing the first frame of a video to be tokenized independently)   
 - Window attention to reduce computing/memory costs   
 - Cascaded pipeline for high-quality generation    
 
![](../../assets/08-93.png) 

Gupta et al., “Photorealistic Video Generation with Diffusion Models,” arXiv 2023.     

P94   
> &#x2753; 已训好的图像生成模型，怎样转成同风格的视频生成模型？    


P95   
## Other Closed-Source Works


|||
|--|--|
| ![](../../assets/08-95-1.png)  | **Latent Shift** (An et al.)<br>Shift latent features for better temporal coherence <br> “Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-2.png) | **Video Factory** (Wang et al.)<br> Modify attention mechanism for better temporal coherence <br> “VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation,” arXiv 2023. |
| ![](../../assets/08-95-3.png) | **PYoCo** (Ge et al.)<br> Generate video frames starting from similar noise patterns <br> “Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models,” ICCV 2023.  |
| ![](../../assets/08-95-4.png)  | **VideoFusion** (Lorem et al.)<br> Decompose noise into shared “base” and individual “residuals”<br>“VideoFusion: ecomposed Diffusion Models for High-Quality Video Generation,” CVPR 2023. |

> &#x2705; Framwork (1) 在原模型中加入 temporal layers (2) fix 原模型，训练新的 layers (3) 把 lager 插入到目标 T2 I 模型中。   


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/