# 3.2 Training-free

P178   
![](../../assets/08-178.png) 

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|117|2023|TokenFlow: Consistent Diffusion Features for Consistent Video Editing|
||2023|FateZero: Fusing Attentions for Zero-shot Text-based Video Editing| Attention map fusing for better temporal consistency <br> - During DDIM inversion, save inverted self-/cross-attention maps <br> - During editing, use some algorithms to blend inverted maps and generated maps   |![](../../assets/08-184.png)   <br>![](../../assets/08-185.png)   <br> ![](../../assets/08-186.png)   |

P187   
## More Works

|||
|--|--|
| ![](../../assets/08-187-1.png)  | **MeDM** (Chu et al.) <br> OpScal flow-based guidance for temporal consistency <br> “MeDM: Mediagng Image Diffusion Models for Video-to Video Translagon with Temporal Correspondence Guidance,” arXiv 2023. |
| ![](../../assets/08-187-2.png) | **Ground-A-Video** (Jeong et al.) <br> Improve temporal consistency via modified attention and optical flow <br> “Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models,” arXiv 2023. |
| ![](../../assets/08-187-3.png)  | **Gen-L-Video** (Lorem et al.) <br> Edit very long videos using existing generators <br> “Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising,” arXiv 2023.  |
| ![](../../assets/08-187-4.png)  | **FLATTEN** (Cong et al.) <br> Optical flow-guided attention for temporal consistency <br> “Flatten: optical flow-guided attention for consistent text-to-video editing,” arXiv 2023. |
| ![](../../assets/08-187-5.png) | **InFusion** (Khandelwal et al.) <br> Improve temporal consistency via fusing latents <br> “InFusion: Inject and Attention Fusion for Multi Concept Zero-Shot Text-based Video Editing,” ICCVW 2023.  |
| ![](../../assets/08-187-6.png)  | **Vid2Vid-Zero** (Wang et al.) <br> Improve temporal consistency via cross￾attention guidance and null-text inversion <br> “Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models,” arXiv 2023. |


P194
> &#x2705; 对于输入文本的每个 wordtoken, 都可以通过 attentior map 找到图像中的大概位置，把要去除的 token mask 掉，剩下部分保留。生成图像则把非 token 部分 mask 掉，以此进行两部分的融合。  

P197
> &#x2705; 基于不同信号的各种版的 control net.   

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/
