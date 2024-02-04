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


![](../assets/D2-25-1.png)     
![](../assets/D2-25-2.png)     
![](../assets/D2-25-3.png)   

Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models", </u> CVPR 2023    

P26   
## Imagic: Text-Based Real Image Editing with Diffusion Models

![](../assets/D2-26.png) 

Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models", </u>CVPR 2023   


P27   
## Prompt-to-Prompt Image Editing with Cross-Attention Control

![](../assets/D2-27.png)

Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control", </u>ICLR 2023    

P28   
## Prompt-to-Prompt Image Editing with Cross￾Attention Control    

![](../assets/D2-28.png)

Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control",</u> ICLR 2023    

P29   
## Prompt-to-Prompt Image Editing with Cross-Attention Control   

![](../assets/D2-29.png)

Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control",</u> ICLR 2023     

P30   
## InstructPix2Pix: Learning to Follow Image Editing Instructions  

![](../assets/D2-30.png)   

Brooks et al., <u>"Instructpix2pix: Learning to follow image editing instructions”,</u> CVPR 2023    

P31   
## InstructPix2Pix: Learning to Follow Image Editing Instructions   

![](../assets/D2-31-1.png)    
![](../assets/D2-31-2.png)    

Brooks et al., <u>"Instructpix2pix: Learning to follow image editing instructions”,</u> CVPR 2023    


P32   
## Personalization with diffusion models   

![](../assets/D2-32.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    

P33   
## DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation    

![](../assets/D2-33.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    

P34   
## The DreamBooth Method  

![](../assets/D2-34.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023     
 
P35   
## DreamBooth Results

![](../assets/D2-35.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023     

P36   
## DreamBooth Applications    

![](../assets/D2-36.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    

P37    
## Textual Inversion: Optimizing Text Embedding   

![](../assets/D2-37.png) 

Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    

P38   
## Textual Inversion: Optimizing Text Embedding  

![](../assets/D2-38.png) 

Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    

P39   
## Textual Inversion Results   

![](../assets/D2-39.png) 

Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    

P40    
## Works well for artistic styles    

![](../assets/D2-40.png) 

Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    

P41   
## Low-rank Adaptation (LoRA)   

 - Lora: Low-rank adaptation of large language models

![](../assets/D2-41.png) 

Lora [Edward J. Hu*, Yelong Shen*, et al., ICLR 2022]     
Lora + Dreambooth (by Simo Ryu): <https://github.com/cloneofsimo/lora>     

P42   
## Low-rank Adaptation (LoRA)

![](../assets/D2-42.png) 

Finetuned with only 9 images Visualized every 500 steps    

Lora [Edward J. Hu*, Yelong Shen*, et al., ICLR 2022]    
Lora + Dreambooth (by Simo Ryu): <https://github.com/cloneofsimo/lora>    

P43
## Fine-tuning all model weights

**Storage requirement**. 4GB storage for each fine-tuned model.    
**Compute requirement**. It requires more VRAM/training time.      
**Compositionality**. Hard to combine multiple models.    


Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P45   
## Analyze change in weights   

![](../assets/D2-45.png) 

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P46   
## Only fine-tune cross-attention layers

![](../assets/D2-46.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P47   
## How to prevent overfitting?    

![](../assets/D2-47.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    


P48   
## Personalized concepts   

![](../assets/D2-48.png)   

How to describe personalized concepts?    

V* dog   

Where V* is a modifier token in the text embedding space    

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P49   
## Personalized concepts

Also fine-tune the modifier token V* that describes the personalized concept   

![](../assets/D2-49.png)   

P50   
## Single concept results

![](../assets/D2-50.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P51   
## Multiple new concepts?    

![](../assets/D2-51.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P52   
## Joint training

1. Combine the training dataset of multiple concepts    

![](../assets/D2-52.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023     

P53   
## Two concept results

![](../assets/D2-53.png)   

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    


P54   
## Two concept results   

![](../assets/D2-54.png) 

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P55   
## Key-Locked Rank One Editing for Text-to-Image Personalization   

![](../assets/D2-55.png)    

Tewel et al., <u>"Key-Locked Rank One Editing for Text-to-Image Personalization",</u> SIGGRAPH 2023    


P56   
## T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models

![](../assets/D2-56-1.png)    

***√ Plug-and-play.*** ***Not affect original network topology and generation ability***   
\\({\color{Orange} \text{√ Simple and small.}  } \\) ***~77M parameters and ~300M storage***    
√ Flexible. ***Various adapters for different control conditions***    
√ Composable. ***Several adapters to achieve multi-condition control***    
√ Generalizable. ***Can be directly used on customed models***    

![](../assets/D2-56-2.png)    

Mou et al., <u>"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models",</u> arXiv 2023   

P57   
## T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models    

![](../assets/D2-57.png)    

Mou et al., <u>"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models",</u> arXiv 2023    

P58   
## Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

![](../assets/D2-58.png) 

Zhang and Agrawala, <u>"Adding Conditional Control to Text-to-Image Diffusion Models",</u> arXiv 2023    

P59   

![](../assets/D2-59.png) 

P60   
## Adding Conditional Control to Text-to-Image Diffusion Models





![](../assets/D2-1.png) 
