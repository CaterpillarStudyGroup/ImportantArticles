P126   
# 2.6 Long video generation
P127  

![](../../assets/08-127.png) 

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|60|2023|Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,”|A “diffusion over diffusion” architecture for very long video generation ||[link](https://caterpillarstudygroup.github.io/ReadPapers/60.html)|
|80|2025|One-Minute Video Generation with Test-Time Training|1. 引入TTT层，通过TTT层动态调整模型隐藏状态，增强对长序列的全局理解能力。<br>2. 通过门控机制防止TTT层训练初期引入噪声。<br>3. 多阶段训练策略：从3秒片段逐步扩展至63秒，仅微调TTT层和门控参数，保留预训练模型的知识。|Test Time Training, RNN, |[link](https://caterpillarstudygroup.github.io/ReadPapers/80.html)|
||2025|Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-free Long Video Diffusion|

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
