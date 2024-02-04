P1   
## Applications of Denoising Diffusion Models on Images

P2   
## Outline

 - **Diffusion model architectures**    
 - **Editing and customization with diffusion models**    
 - **Other applications of diffusion models**    

P4   
## Architecture


P5   
## U-Net Architecture

![](../assets/D2-5-1.png) 

![](../assets/D2-5-2.png) 

Ronneberger et al., <u>“U-Net: Convolutional Networks for Biomedical Image Segmentation”, </u>MICCAI 2015      
Rombach et al., <u>"High-Resolution Image Synthesis with Latent Diffusion Models",</u> CVPR 2022    

P6   
## U-Net Architecture

**Imagen** aharia et al.   
**Stable Diffusion** Rombach et al.    
**eDiff-I** Balaji et al.    
 
Saharia et al. <u>“Photorealistic text-to-image diffusion models with deep language understanding”, </u>NeurIPS 2022    
Rombach et al., <u>"High-Resolution Image Synthesis with Latent Diffusion Models", </u>CVPR 2022    
Balaji et al.,” <u>ediffi: Text-to-image diffusion models with an ensemble of expert denoisers”, </u>arXiv 2022    

P7    
## Transformer Architecture

![](../assets/D2-7-1.png) 

![](../assets/D2-7-2.png) 

Dosovitskiy et al., <u>“An image is worth 16x16 words: Transformers for image recognition at scale”, </u>ICLR 2021     
Bao et al.,<u> "All are Worth Words: a ViT Backbone for Score-based Diffusion Models", </u>arXiv 2022    

P8   
## Transformer Architecture

Peebles and Xie, <u>"Scalable Diffusion Models with Transformers", </u>arXiv 2022    
Bao et al., <u>"One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale", </u>arXiv 2023    
Hoogeboom et al., <u>"simple diffusion: End-to-end diffusion for high resolution images", </u>arXiv 2023    


P9    
## Image editing and customization with diffusion models

P10    
## Image editing and customization with diffusion models
 - RGB pixel guidance   
 - Text guidance   
 - Reference image guidance   

P11   
## How to perform guided synthesis/editing?   

![](../assets/D2-11.png) 

P12   
## SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations   

First perturb the input with **Gaussian noise** and then progressively remove the noise using a pretrained diffusion model.    

![](../assets/D2-12.png) 

Gradually projects the input to the manifold of natural images.

Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", </u>ICLR 2022   

P13   
## Fine-grained control using strokes

![](../assets/D2-13.png) 

Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", </u>ICLR 2022    

P16   
## Image compositing  

![](../assets/D2-16.png) 

Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", </u>ICLR 2022    

P17   
## Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

![](../assets/D2-17.png) 

Li et al., <u>"Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models", </u>NeurIPS 2022    

P18   
## DDIM Inversion

![](../assets/D2-18.png) 

Song et al., <u>"Denoising Diffusion Implicit Models", </u>ICLR 2021    

P19   
## Style transfer with DDIM inversion

![](../assets/D2-19.png) 

Su et al., <u>"Dual diffusion implicit bridges for image-to-image translation", </u>ICLR 2023    

P20   
## Style transfer with DDIM inversion

![](../assets/D2-20.png) 

Su et al., <u>"Dual diffusion implicit bridges for image-to-image translation",</u> ICLR 2023    

P21    
## DiffEdit: Diffusion-based semantic image editing with mask guidance    

Instead of asking users to provide the mask, the model will generate the mask itself based on the caption and query.    

![](../assets/D2-21.png) 

Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance", </u>ICLR 2023    

P22   
## DiffEdit: Diffusion-based semantic image editing with mask guidance    

![](../assets/D2-22.png) 

Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance",</u> ICLR 2023   

P23   
## DiffEdit: Diffusion-based semantic image editing with mask guidance

![](../assets/D2-23.png) 

Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance", </u>ICLR 2023    

P24   
## Imagic: Text-Based Real Image Editing with Diffusion Models   

![](../assets/D2-24.png) 

Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models",</u> CVPR 2023     

P25   
## Imagic: Text-Based Real Image Editing with Diffusion Models






![](../assets/D2-1.png) 
