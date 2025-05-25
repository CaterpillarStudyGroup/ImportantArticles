![](../../assets/08-30.png)

P36  

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023.11|Stablevideo: Text-driven consistency-aware diffusion video editing|
|57|2023.9|Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation|直接在像素空间实现时序扩散模型，结合修复（inpainting）与超分辨率技术生成高分辨率视频||[link](https://caterpillarstudygroup.github.io/ReadPapers/57.html)| 
||2023.8|I2vgen-xl: High-quality image-to-video|提出级联网络，通过分离内容与运动因素提升模型性能，并利用静态图像作为引导增强数据对齐。|
|48|2023.4|Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models|首次将潜在扩散模型（LDM）范式引入视频生成，在潜在空间中加入时序维度<br>T2I(LDM) -> T2V(SVD)<br>Cascaded generation|Video LDM|[link](https://caterpillarstudygroup.github.io/ReadPapers/48.html)| 
|59|2023|AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning|| |[link](https://caterpillarstudygroup.github.io/ReadPapers/59.html)|
||2023|Chen et al., “GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation,”|Transformer-based diffusion for text-to-video generation<br> &#x2705;Transformer-based architecture extended from DiT (class-conditioned transformer-based LDM) <br> &#x2705;Train T2I \\(\to \\)  insert temporal self-attn \\(\to \\) joint image-video finetuning (motion-free guidance)    |![](../../assets/08-91.png) |
||2023|Gupta et al., “Photorealistic Video Generation with Diffusion Models,”|Transformer-based diffusion for text-to-video generation<br> &#x2705;Transformer-based denoising diffusion backbone<br> &#x2705;Joint image-video training via unified image/video latent space (created by a joint 3D encoder with causal 3D conv layers, allowing the first frame of a video to be tokenized independently)<br> &#x2705;Window attention to reduce computing/memory costs<br> &#x2705;Cascaded pipeline for high-quality generation   | ![](../../assets/08-93.png) |
||2022.11|Imagen Video: High Definition Video Generation with Diffusion Models|提出级联扩散模型以生成高清视频，并尝试将文本到图像（text-to-image）范式迁移至视频生成<br>级联扩散模型实现高清生成，质量与分辨率提升<br> &#x2705; 先在 image 上做 cascade 生成 <br> &#x2705; 视频是在图像上增加时间维度的超分   <br> &#x2705; 每次的超分都是独立的 diffusion model?   <br> 7 cascade models in total.  <br> 1 Base model (16x40x24) <br> 3 Temporal super-resolution models. <br> 3 Spatial super-resolution models. <br> &#x2705; 通过 7 次 cascade，逐步提升顺率和像素的分辨率，每一步的训练对上一步是依赖的。   |<br>Cascade| ![](../../assets/08-63-1.png) <br> ![](../../assets/08-63-2.png)<br>![](../../assets/D3-52.png)  |
|56|2022.9|Make-A-Video: Text-to-Video Generation without Text-Video Data|||[link](https://caterpillarstudygroup.github.io/ReadPapers/56.html)|
|55|2022.4|Video Diffusion Models|首次采用3D U-Net结构的扩散模型预测并生成视频序列<br>引入conv(2+1)D，temporal attention||[link](https://caterpillarstudygroup.github.io/ReadPapers/55.html)|

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/