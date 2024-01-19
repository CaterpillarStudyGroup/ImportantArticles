
P177  
# 3 Video Editing

## 3.2 Training-free

P178   
![](../../assets/08-178.png) 

> &#x2705; 视频编辑领域比较难的问题：怎么保持时序一致性。   

P179   
## TokenFlow

Consistent high-quality semantic edits

Main challenge using T2I to edit videos without finetuning: temporal consistency  

![](../../assets/08-179.png) 

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Edigng,” arXiv 2023.    

P180   
## TokenFlow

Consistent high-quality semantic edits

**Key Idea**

 - Achieve consistency by enforcing the inter-frame correspondences in the original video   

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Edigng,” arXiv 2023.     

> &#x2705; 在 UNet 中抽出 feature map 之后，找 corresponden 并记录下来。在 denoise 过程中把这个 correspondence 应用起来。   

P181   
## TokenFlow   

Consistent high-quality semanec edits

**Main idea**

![](../../assets/08-181.png) 

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.    

P182   
## TokenFlow

Consistent high-quality semantic edits

**Main idea**

During conditional denoising, use features from corresponding positions in preceding and following frames instead of the pixel's own feature at output of extended-attention

|||
|--|--|
| ![](../../assets/08-182.png)  | ![](../../assets/08-182-1.png)  |


Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.    

P183   
## TokenFlow

Consistent high-quality semantic edits

![](../../assets/08-183.png) 

  
Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.   

> &#x2705; 在 DDIM inversion 过程中，把 attention maps 保存下来了，在 denoise 时，把这个 map 结合进去。    


P184   
## FateZero   

Attention map fusing for better temporal consistency

**Methodology** 

 - During DDIM inversion, save inverted self-/cross-attention maps    
 - During editing, use some algorithms to blend inverted maps and generated maps   

![](../../assets/08-184.png) 

Qi et al., “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing,” ICCV 2023.    

P185   
## FateZero   

Attention map fusing for better temporal consistency

**Methodology**

 - During DDIM inversion, save inverted self-/cross-avenkon maps   
 - During edikng, use some algorithms to blend inverted maps and generated maps   

![](../../assets/08-185.png) 

Qi et al., “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing,” ICCV 2023.    


P186    
## FateZero  

Attention map fusing for better temporal consistency

![](../../assets/08-186.png) 

Qi et al., “FateZero: Fusing Akengons for Zero-shot Text-based Video Edigng,” ICCV 2023.    

P187   
## Training-Free Video Editing: More Works


|||
|--|--|
| ![](../../assets/08-187-1.png)  | **MeDM** (Chu et al.) <br> OpScal flow-based guidance for temporal consistency <br> “MeDM: Mediagng Image Diffusion Models for Video-to Video Translagon with Temporal Correspondence Guidance,” arXiv 2023. |
| ![](../../assets/08-187-2.png) | **Ground-A-Video** (Jeong et al.) <br> Improve temporal consistency via modified attention and optical flow <br> “Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models,” arXiv 2023. |
| ![](../../assets/08-187-3.png)  | **Gen-L-Video** (Lorem et al.) <br> Edit very long videos using existing generators <br> “Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising,” arXiv 2023.  |
| ![](../../assets/08-187-4.png)  | **FLATTEN** (Cong et al.) <br> Optical flow-guided attention for temporal consistency <br> “Flatten: optical flow-guided attention for consistent text-to-video editing,” arXiv 2023. |
| ![](../../assets/08-187-5.png) | **InFusion** (Khandelwal et al.) <br> Improve temporal consistency via fusing latents <br> “InFusion: Inject and Attention Fusion for Multi Concept Zero-Shot Text-based Video Editing,” ICCVW 2023.  |
| ![](../../assets/08-187-6.png)  | **Vid2Vid-Zero** (Wang et al.) <br> Improve temporal consistency via cross￾attention guidance and null-text inversion <br> “Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models,” arXiv 2023. |
