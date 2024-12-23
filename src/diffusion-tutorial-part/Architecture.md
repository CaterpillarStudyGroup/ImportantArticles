# Architecture

P5   
## U-Net Based Diffusion Architecture

### U-Net Architecture

![](../assets/D2-5-1.png) 

> &#x2705; U-Net的是Large Scale Image Diffusion Model中最常用的backbone。  

> &#x1F50E; Ronneberger et al., <u>“U-Net: Convolutional Networks for Biomedical Image Segmentation”, </u>MICCAI 2015    

### Pipeline 

![](../assets/D2-5-2.png) 

> &#x2705; 包含Input、U-Net backbone、Condition。  
> &#x2705; Condition 通常用 Concat 或 Cross attention 的方式与 Content 相结合。    

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|45|2022|High-Resolution Image Synthesis with Latent Diffusion Models|**Stable Diffusion**, U-Net Based Diffusion Architecture<br>&#x2705; (1)：在 latent space 上工作<br> &#x2705; (2)：引入多种 condition．|UNet|[link](https://caterpillarstudygroup.github.io/ReadPapers/45.html)|
||2022|Photorealistic text-to-image diffusion models with deep language understanding|Imagen|
||2022|ediffi: Text-to-image diffusion models with an ensemble of expert denoiser|eDiff-I|

P7    
## Transformer Architecture

### Vision Transformer(ViT)

![](../assets/D2-7-1.png) 

Dosovitskiy et al., <u>“An image is worth 16x16 words: Transformers for image recognition at scale”, </u>ICLR 2021    

### Pipeline

![](../assets/D2-7-2.png) 

> &#x2705; 特点：  
> &#x2705; 1. 把 image patches 当作 token.    
> &#x2705; 2. 在 Shallow layer 与 deep layer 之间引入 long skip connection.    

Bao et al.,<u> "All are Worth Words: a ViT Backbone for Score-based Diffusion Models", </u>arXiv 2022    


P8   
### Application
|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2022|Scalable Diffusion Models with Transformers|    
||2023|One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale|
||2023|simple diffusion: End-to-end diffusion for high resolution images|
