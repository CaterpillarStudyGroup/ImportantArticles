# Tutorial: Video Diffusion Models

**Mike Shou**   

Asst Prof, National U. of Singapore   

Joint work with Pei Yang & Jay Wu   

Slides:<https://sites.google.com/view/showlab/tutorial> 

![](./assets/08-001.png)


P2  
## Video Foundation Model  

![](./assets/08-01.png)


P5  
## Outline of “Tutorial: Video Diffusion Models”

1. Fundamentals of Diffusion Models
2. **Video Generation**
3. **Video Editing**
4. Summary


P6  
# 1 DDPM (Denoising Diffusion Probabilistic Models)


P7  
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-02.png)

Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.   
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.   
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.   
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.   


P8   
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-03.png)

Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.    
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.    
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.   
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.   


P9  
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-04.png)

Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.    
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.   
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.   
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.   


P10   
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-05.png)

Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.


P11   
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-11.png)


Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.    
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.    
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.    
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.   

P12   
## DDPM (Denoising Diffusion Probabilistic Models)

![](./assets/08-12.png)

Ho et al., “Denoising Diffusion Probabilistic Models,” NeurIPS 2020.   
Sohl-Dickstein et al., “Deep Unsupervised Learning using Nonequilibrium Thermodynamics,” ICML 2015.   
Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.   
Vahdat et al., “Denoising Diffusion Models: A Generative Learning Big Bang,” CVPR 2023 Tutorial.   

P14  

## DDIM (Denoising Diffusion Implicit Models)

![](./assets/08-14.png)

Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.    
Song et all, “Denoising Diffusion Implicit Models,” ICLR 2021.   

P15   
## Denoising Diffusion Models

DDPM vs DDIM   

|||
|--|--|
| ![](./assets/08-15-1.png) | **DDPM cannot skip timesteps**  <br> A few hundreds steps to generate an image |
|![](./assets/08-15-2.png) |**DDIM can skip timesteps** <br> Say 50 steps to generate an image |

Song et al., “Score-Based Generative Modeling through Stochastic Differential Equations,” ICLR 2021.   
Song et all, “Denoising Diffusion Implicit Models,” ICLR 2021.    

P16   
## DDIM Inversion

The task of Inversion

![](./assets/08-16.png)

Song et al., “Denoising Diffusion Implicit Models,” ICLR 2021.    
Su et al., “Dual Diffusion Implicit Bridges for Image-to-Image Translation,” ICLR 2023.    
Mokadi et al., “Null-text Inversion for Editing Real Images using Guided Diffusion Models,” CVPR 2023.    


P17   
## DDIM Inversion

Based on the assumption that the ODE process can be reversed in the limit of small steps    

Forward Diffusion Process: Add \\(\mathcal{N} (0,\mathbf{I} ) \\) Noise

DDIM Inversion Process: Add Noise **inverted** by the trained DDIM denoiser

Song et al., “Denoising Diffusion Implicit Models,” ICLR 2021.    
Su et al., “Dual Diffusion Implicit Bridges for Image-to-Image Translation,” ICLR 2023.   
Mokadi et al., “Null-text Inversion for Editing Real Images using Guided Diffusion Models,” CVPR 2023.    


p18   
## Wanted to learn more?

 - CVPR Tutorial (English): <https://www.youtube.com/watch?v=cS6JQpEY9cs>   
 - Lil’s blog: <https://lilianweng.github.io/posts/2021-07-11-diffusion-models/>   
 - Hung-yi Lee (Chinese):    
    - <https://www.youtube.com/watch?v=azBugJzmz-o>   
    - <https://www.youtube.com/watch?v=ifCDXFdeaaM>   
 - Checkout codes -- Always associate theory and implementation!   


P20   
## CLIP   

Encoders bridge vision and language


 - CLIP text-/image-embeddings are commonly used in diffusion models for conditional generation

|||
|--|--|
| ![](./assets/08-20-1.png)  |  ![](./assets/08-20-2.png) |

Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.     


P21   

## Latent Diffusion

![](./assets/08-21.png) 

P22   
## Stable Diffusion

Conditional/unconditional image generation    

![](./assets/08-22.png) 

Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR 2022.      


P24   
## LoRA: Low-Rank Adaptation   

Few-shot finetuning of large models for personalized generation

|||
|--|--|
| ![](./assets/08-24-1.png)  |  ![](./assets/08-24-2.png) |


Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” arXiv 2021.    
Gu et al., “Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models,” arXiv 2023.   


P25   
## DreamBooth   

Few-shot finetuning of large models for generating personalized concepts

|||
|--|--|
| ![](./assets/08-25-1.png)  |  ![](./assets/08-25-2.png) |

Ruiz et al., “DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation,” CVPR 2023.    


P26  
## ControlNet    

Conditional generation with various guidances    

![](./assets/08-26.png)

Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV 2023.     


P27   
## ControlNet   

Conditional generation with various guidances   
 - Finetune parameters of a trainable copy   

![](./assets/08-27.png)

Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV 2023.    


P28   
## ControlNet   

Conditional generation with various guidances

![](./assets/08-28.png)

Zhang et al., “Adding Conditional Control to Text-to-Image Diffusion Models,” ICCV 2023.    


P29   
# 2 Video Generation

P30  
## Video Foundation Model

![](./assets/08-30.png)


P31  

![](./assets/08-31.png)


P32   

# 2 Video Generation

## 2.1 Pioneering/early works

![](./assets/08-33.png)


P34  
## Problem Definition

**Text-Guided Video Generation**   

Text prompt → video   

Video from Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.    

P35   
## Problem Definition   

![](./assets/08-35.png)


P36  
## Video Diffusion Models  

Recap 3D Conv

|||
|--|--|
| ![](./assets/08-36-1.png)  |  ![](./assets/08-36-2.png) |

Du et al., “Learning Spatiotemporal Features with 3D Convolutional Networks,” ICCV 2015.     

P37  
## Video Diffusion Models

Recap (2+1)D Conv

|||
|--|--|
| ![](./assets/08-37-1.png)  |  ![](./assets/08-37-2.png) |

Du et al., “A Closer Look at Spatiotemporal Convolutions for Action Recognition,” CVPR 2018.    

P38   
## Video Diffusion Models   

Early work on video generation

Ho et al., “Video Diffusion Models,” NeurIPS 2022.  


P39   
## Video Diffusion Models

Early work on video generation   

 - 3D U-Net factorized over space and time   
 - Image 2D conv inflated as → space-only 3D conv, i.e., 2 in (2+1)D Conv   
    - Kernel size: (3×3) → (1×3×3)   
    - Feature vectors: (height × weight × channel) → (frame × height × width × channel)   
 - Spatial attention: remain the same   
 - Insert temporal attention layer: attend across the temporal dimension (spatial axes as batch)   

![](./assets/08-39.png) 

Ho et al., “Video Diffusion Models,” NeurIPS 2022.  


P40  
## Make-A-Video

Cascaded generation

![](./assets/08-40.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022. 

P41   
## Make-A-Video

Cascaded generation

![](./assets/08-41.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.     

P42  
## Make-A-Video

Cascaded generation

![](./assets/08-42.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.    


P43   
## Make-A-Video

Cascaded generation

![](./assets/08-43.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.   


P44  
## Make-A-Video

Cascaded generation

**Training**
 - 4 main networks (decoder + interpolation + 2 super-res)   
    - First trained on images alone    
    - Insert and finetune temporal layers on videos   
 - Train on WebVid-10M and 10M subset from HD-VILA-100M   


P45   
## Datasets

The WebVid-10M Dataset

![](./assets/08-45.png) 

Bain et al., “Frozen in Time: A Joint Video and Image Encoder for End to End Paper,” ICCV 2021.    

P46   
## Evaluation Metrics

![](./assets/08-46.png) 


P47   

## Evaluation Metrics   

Quantitative evaluations

**Image-level Evaluation Metrics**

 - Fréchet Inception Distance (FID, ↓): semantic similarity between images   
 - Peak Signal-to-Noise Ratio (PSNR, ↑): pixel-level similarity between images   
 - Structural Similarity Index (SSIM, ↓): pixel-level similarity between images   
 - CLIPSIM (↑): image-text relevance   

**Video-level Evaluation Metrics**

 - Fréchet Video Distance (FVD, ↓): semantic similarity & temporal coherence   
 - Kernel Video Distance (KVD, ↓): video quality (via semantic features and MMD)   
 - Video Inception Score (IS, ↑): video quality and diversity   
 - Frame Consistency CLIP Score (↑): frame temporal semantic consistency   


P48   
## Fréchet Inception Distance (FID)

Semantic similarity between images

![](./assets/08-48.png) 

**Lantern image generated with Stable Diffusion 2.1.**    

Heusel et al., “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium,” NeurIPS 2017.    
Hung-Yi Lee, “Machine Learning 2023 Spring,” National Taiwan University.    

P49   
## Peak Signal-to-Noise Ratio (PSNR)

Pixel-level similarity between images

 - For two images \\(x,y \text{ of shape }  M\times N\\):   

\begin{align*} \mathrm{PSNR} (x,y) = 10 \log_{10}{} \frac{255^2}{\mathrm{MSE} (x,y)}  \end{align*}

where    

\begin{align*} \mathrm{MSE} (x,y) = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} (x_{ij}-y_{ij})^2\end{align*}

Horé et al., “Image Quality Metrics: PSNR vs. SSIM,” ICPR 2010.    

P50  
## Structural Similarity Index Measure (SSIM)

Pixel-level similarity between images

 - Model any image distortion as a combination of:   
(1) loss of correlation, (2) luminance distortion, (3) contrast distortion

 - For two images \\(x,y \text{ of shape }  M\times N\\):   

\begin{align*}  \mathrm{SSIM} (x,y)=l(x,y)\cdot c(x,y)\cdot s(x,y)\end{align*}

where   

\begin{align*} \begin{cases}
 \text{Lumiannce Comparison Funckon:} l(x,y)=\frac{2\mu _x\mu _y+C_1}{\mu _x^2+\mu _y^2+C_1}  \\\\ 
 \text{Contrast Comparison Funckon:} c(x,y)=\frac{2\sigma  _x\sigma  _y+C_2}{\sigma  _x^2+\sigma  _y^2+C_2}  \\\\ 
  \text{Structure Comparison Funckon:} s(x,y)=\frac{\sigma  _{xy}+C_3}{\sigma  _{x}\sigma  _{y}+C_3}  \end{cases}\end{align*}
 

Wang et al., “Image Quality Assessment: from Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, April 2004.   
Horé et al., “Image Quality Metrics: PSNR vs. SSIM,” ICPR 2010.   

P51   
## CLIP Similarity   

Image-caption similarity

![](./assets/08-51.png) 

Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.      

P52   
## Fréchet Video Distance (FVD)  

Semantic similarity and temporal coherence between two videos    

![](./assets/08-52.png) 

Unterthiner et al., “FVD: A new Metric for Video Generation,” ICLR 2019.    
Unterthiner et al., “Towards Accurate Generative Models of Video: A New Metric & Challenges,” arXiv 2018.     


P53   
## Kernel Video Distance   

Video quality assessment via semantic features and MMD  

![](./assets/08-53.png) 

Unterthiner et al., “FVD: A new Metric for Video Generation,” ICLR 2019.    
Unterthiner et al., “Towards Accurate Generative Models of Video: A New Metric & Challenges,” arXiv 2018.      


P54   
## Video Inception Score (IS)

Video quality and diversity

![](./assets/08-54.png) 

Salimans et al., “Improved Techniques for Training GANs,” NeurIPS 2016.    
Barratt et al., “A Note on the Inception Score,” ICML 2018.    
Saito et al., “Train Sparsely, Generated Densely: Memory-Efficient Unsupervised Training of High-Resolution Temporal GAN,” IJCV 2020. 


P55   
## Frame Consistence CLIP scores

Frame temporal semantic consistency


 - Compute CLIP image embeddings for all frames   
 - Report average cosine similarity between all pairs of frames   

![](./assets/08-55.png) 

Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” ICML 2021.    


P57   
## Evaluation Metrics    

Hybrid evaluation

**EvalCrafter**

 - Creates a balanced prompt list for evaluation   
 - **Multi-criteria decision analysis** on 18 metrics: visual quality, content quality…   
 - Regress the coefficients of all metrics to generate an overall score aligned with user opinions   

![](./assets/08-57.png) 

Liu et al., “EvalCrafter: Benchmarking and Evaluating Large Video Generation Models,” arXiv 2023.      

P58  
## Make-A-Video   

Cascaded generation

![](./assets/08-58.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.    


P59   
## Make-A-Video   

Cascaded generation

![](./assets/08-59.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.     


P60  
## Make-A-Video   

Cascaded generation

![](./assets/08-60.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.    


P62   
## Make-A-Video   

Cascaded generation

From static to magic   
Add motion to a single image or fill-in the in-betw    

![](./assets/08-62.png) 

Singer et al., “Make-A-Video: Text-to-Video Generation without Text-Video Data,” arXiv 2022.    


P63   
## Imagen & Imagen Video

Leverage pretrained T2I models for video generation; Cascaded generation

|||
|--|--|
| ![](./assets/08-63-1.png)  |  ![](./assets/08-63-2.png) |


Imagen: Saharia et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding,” arXiv 2022.    
Imagen Video: Ho et al., “Imagen Video: High Definition Video Generation with Diffusion Models,” arXiv 2022.    


P64   
## Align your Latents

Leverage pretrained T2I models for video generation; 
Cascaded generation   

![](./assets/08-64.png)

Blattmann et al., “Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models,” CVPR 2023.     


P65   
## Align your Latents   

Leverage pretrained T2I models for video generation

**Inserting Temporal Layers**   

 - Latent space diffusion model: insert temporal convolutional & 3D attention layers   
 - Decoder: add 3D convolutional layers   
 - Upsampler diffusion model: add 3D convolution layers   

Blattmann et al., “Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models,” CVPR 2023.     

P66   
# 2 Video Generation   

## 2.2 Open-source base models


P67   
![](./assets/08-67.png)

P68   
## ModelScopeT2V

Leverage pretrained T2I models for video generation

 - Inflate Stable Diffusion to a 3D model, preserving pretrained weights   
 - Insert spatio-temporal blocks, can handle varying number of frames   

|||
|--|--|
| ![](./assets/08-68-1.png)  |  ![](./assets/08-68-2.png) |


Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.    

P69   
## ModelScopeT2V

Leverage pretrained T2I models for video generation   

 - Inflate Stable Diffusion to a 3D model, preserving pretrained weights    
 - Insert spatio-temporal blocks   

![](./assets/08-69.png) 


Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.     


P70   
## ModelScopeT2V    

Leverage pretrained T2I models for video generation


 - Inflate Stable Diffusion to a 3D model, preserving pretrained weights   
 - Insert spatio-temporal blocks, **can handle varying number of frames**   

![](./assets/08-70.png) 

Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023. 


P71   
## ModelScopeT2V

Length = 1   
Model generate images   

P72   
## ModelScopeT2V   

Leverage pretrained T2I models for video generation

![](./assets/08-72.png) 


ZeroScope: finetunes ModelScope on a small set of high-quality videos, resulting into higher resolution at 
1024 x 576, without the Shutterstock watermark    

Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.     


P73   
## ModelScopeT2V   

Leverage pretrained T2I models for video generation  


Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.  

P74  
## ModelScopeT2V

Leverage pretrained T2I models for video generation

|||||
|--|--|--|--|
| " Robot dancing in times square,” arXiv 2023.  | " Clown fish swimming through the coral reef,” arXiv 2023.| " Melting ice cream dripping down the cone,” arXiv 2023.| " Hyper-realistic photo of an abandoned industrial site during a storm,” arXiv 2023.|
| ![](./assets/08-74-1.png)  |  ![](./assets/08-74-2.png) | ![](./assets/08-74-3.png)  |  ![](./assets/08-74-4.png) |


Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.    


P75   
## Show-1

Better text-video alignment? Generation in both pixel- and latent-domain

![](./assets/08-75.png) 

Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.    

P76   
## Show-1

Better text-video alignment? Generation in both pixel- and latent-domain   

**Motivation**

 - Pixel-based VDM achieves better text-video alignment than latent-based VDM   

|||
|--|--|
| ![](./assets/08-76-1.png) | ![](./assets/08-76-2.png) |


Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.   

P77   
## Show-1

Generation in both pixel- and latent-domain

**Motivation**

 - Pixel-based VDM achieves better text-video alignment than latent-based VDM   
 - Pixel-based VDM takes much larger memory than latent-based VDM    

![](./assets/08-77.png) 


Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.   


P78   
## Show-1   

Generation in both pixel- and latent-domain  

**Motivation** 

 - Use Pixel-based VDM in low-res stage   
 - Use latent-based VDM in high-res stage   

![](./assets/08-78.png) 

Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.   


P79   
## Show-1   

Generation in both pixel- and latent-domain

<https://github.com/showlab/Show-1>

 - Better text-video alignment   
 - Can synthesize large motion   
 - Memory-efficient   

Zhang et al., “Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation,” arXiv 2023.   


P80  
## VideoCrafter  

• Latent diffusion inserted with temporal layers

![](./assets/08-80.png) 

Chen et al., “VideoCrafter1: Open Diffusion Models for High-Quality Video Generation,” arXiv 2023.    

P81  
## LaVie  

Joint image-video finetuning with curriculum learning

![](./assets/08-81.png) 

Wang et al., “LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models,” arXiv 2023.   

P83   
## Stable Video Diffusion  

Scaling latent video diffusion models to large datasets

**Data Processing and Annotation** 

 - Cut Detection and Clipping    
    - Detect cuts/transitions at multiple FPS levels   
    - Extract clips precisely using keyframe timestamps   
 - Synthetic Captioning   
    - Use CoCa image captioner to caption the mid-frame of each clip   
    - Use V-BLIP to obtain video-based caption   
    - Use LLM to summarise the image- and video-based caption   
    - Compute CLIP similarities and aesthetic scores
 - Filter Static Scene   
    - Use dense optical flow magnitudes to filter static scenes   
 - Text Detection   
    - Use OCR to detect and remove clips with excess text    


Blattmann et al., “Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets,” 2023.     

P84   
## Stable Video Diffusion   

Scaling latent video diffusion models to large datasets

**Data Processing and Annotation**  



Blattmann et al., “Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets,” 2023. 84

![](./assets/08-84.png) 

P85  
## Stable Video Diffusion   

Scaling latent video diffusion models to large datasets

**Stage I: Image Pretraining**

 - Initialize weights from Stable Diffusion 2.1 (text-to-image model)   

![](./assets/08-85.png) 

Blattmann et al., “Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets,” 2023.    

P86   
## Stable Video Diffusion   

Scaling latent video diffusion models to large datasets

**Stage II: Curating a Video Pretraining Dataset**

 - Systematic Data Curation
    - Curate subsets filtered by various criteria (CLIP-, OCR-, optical flow-, aesthetic-scores…)
    - Assess human preferences on models trained on different subsets
    - Choose optimal filtering thresholds via Elo rankings for human preference votes
 - Well-curated beats un-curated pretraining dataset

![](./assets/08-86.png) 

Blattmann et al., “Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets,” 2023.  

P87   
## Stable Video Diffusion  

Scaling latent video diffusion models to large datasets

**Stage III: High-Quality Finetuning**

 - Finetune base model (pretrained from Stages I-II) on high-quality video data   
    - High-Resolution Text-to-Video Generation   
       - ~1M samples. Finetune for 50K iterations at 576x1024 (in contrast to 320x576 base resolution)   
    - High Resolution Image-to-Video Generation   
    - Frame Interpolation   
    - Multi-View Generation   
 - Performance gains from curation persists after finetuning   

![](./assets/08-87.png) 

Blattmann et al., “Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets,” 2023.    

P89   
# 2 Video Generation

## 2.3 Other closed-source works


P90   

![](./assets/08-90.png) 


P91   
## GenTron

Transformer-based diffusion for text-to-video generation

 - Transformer-based architecture extended from DiT (class-conditioned transformer-based LDM)   
 - Train T2I à insert temporal self-attn à joint image-video finetuning (motion-free guidance)    

![](./assets/08-91.png) 

Chen et al., “GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation,” arXiv 2023.    

P93   
## W.A.L.T.

Transformer-based diffusion for text-to-video generation  

 - Transformer-based denoising diffusion backbone   
 - Joint image-video training via unified image/video latent space (created by a joint 3D encoder with causal 3D conv layers, allowing the first frame of a video to be tokenized independently)   
 - Window attention to reduce computing/memory costs   
 - Cascaded pipeline for high-quality generation    
 
![](./assets/08-93.png) 

Gupta et al., “Photorealistic Video Generation with Diffusion Models,” arXiv 2023.     


P95   
## Other Closed-Source Works


|||
|--|--|
| ![](./assets/08-95-1.png)  | **Latent Shift** (An et al.)<br>Shift latent features for better temporal coherence <br> “Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation,” arXiv 2023. |
| ![](./assets/08-95-2.png) | **Video Factory** (Wang et al.)<br> Modify attention mechanism for better temporal coherence <br> “VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation,” arXiv 2023. |
| ![](./assets/08-95-3.png) | **PYoCo** (Ge et al.)<br> Generate video frames starting from similar noise patterns <br> “Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models,” ICCV 2023.  |
| ![](./assets/08-95-4.png)  | **VideoFusion** (Lorem et al.)<br> Decompose noise into shared “base” and individual “residuals”<br>“VideoFusion: ecomposed Diffusion Models for High-Quality Video Generation,” CVPR 2023. |

P96  
# 2 Video Generation

## 2.4 Training-efficient techniques

P97  

![](./assets/08-97.png) 

P98 
## AnimateDiff  

Transform domain-specific T2I models to T2V models

 - Domain-specific (personalized) models are widely available for image   
    - Domain-specific finetuning methodologies: LoRA, DreamBooth…   
    - Communities: Hugging Face, CivitAI…   
 - Task: turn these image models into T2V models, without specific finetuning   


Guo et al., “AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning,” arXiv 2023.     

P99  
## AnimateDiff   

Transform domain-specific T2I models to T2V models

**Methodology**

 - Train a motion modeling module (some temporal layers) together with frozen base T2I model   
 - Plug it into a domain-specific T2I model during inference   

![](./assets/08-99.png) 

Guo et al., “AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning,” arXiv 2023.    

P100 
## AnimateDiff   

Transform domain-specific T2I models to T2V models

**Methodology** 

 - Train a motion modeling module (some temporal layers) together with frozen base T2I model   
 - Plug it into a domain-specific T2I model during inference   

![](./assets/08-100.png)   

 - Train on WebVid-10M, resized at 256x256 (experiments show can generalize to higher res.)   

Guo et al., “AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning,” arXiv 2023.    

P102  
## Text2Video-Zero   

Use Stable Diffusion to generate videos without any finetuning

**Motivation: How to use Stable Diffusion for video generation without finetuning?**  

 - Start from noises of similar pattern   
 - Make intermediate features of different frames to be similar   


Khachatryan et al., “Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators,” arXiv 2023.    

P103   
## Text2Video-Zero   

Use Stable Diffusion to generate videos without any finetuning

 - Start from noises of similar pattern: given the first frame’s noise, define a global scene motion, used to translate the first frame’s noise to generate similar initial noise for other frames   

![](./assets/08-103.png) 

Khachatryan et al., “Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators,” arXiv 2023.  

P104   
## Text2Video-Zero

Use Stable Diffusion to generate videos without any finetuning

 - Make intermediate features of different frames to be similar: always use K and V from the first frame in self-attention   

![](./assets/08-104.png) 

Khachatryan et al., “Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators,” arXiv 2023.    

P105   
## Text2Video-Zero

Use Stable Diffusion to generate videos without any finetuning

 - Optional background smoothing: regenerate the background, average with the first frame

![](./assets/08-105.png) 

Khachatryan et al., “Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators,” arXiv 2023.   

P107 
## Training Efficient Techniques: More Works


|||
|--|--|
| ![](./assets/08-107-1.png)  | **MagicVideo** (Zhou et al.) <br> Insert causal attention to Stable Diffusion for better temporal coherence <br> “MagicVideo: Efficient Video Generation With Latent Diffusion Models,” arXiv 2022.  |
|  ![](./assets/08-107-2.png)  | **Simple Diffusion Adapter** (Xing et al.) <br> Insert lightweight adapters to T2I models, shift latents, and finetune adapters on videos <br>“SimDA: Simple Diffusion Adapter for Efficient Video Generation,” arXiv 2023. |
| ![](./assets/08-107-3.png) | **Dual-Stream Diffusion Net** (Liu et al.) <br> Leverage multiple T2I networks for T2V <br> “Dual-Stream Diffusion Net for Text-to-Video Generation,” arXiv 2023. |



P108  
# 2 Video Generation

## 2.5 Storyboard
 
P109  
![](./assets/08-109.png) 

P110 
## opyright©Mike Shou, NUS

Human can imagine what does the scene look like “roughly”

“Two men stand in the airport waiting room and stare at the 
airplane thru window”   

What is in your mind now?

Storyboard image from deviantart.com.


P111
## What is a storyboard?

A concept in film production

![](./assets/08-111.png) 

 - Rough sketches/drawings with notes    
 - Example: Inception by Christopher Nola   

Storyboard image from deviantart.com.    


P112 
## What is a storyboard?

A concept in film production

 - How to generate such a storyboard?    
 - As humans, over the years, we have acquired such “visual prior” about object location, object shape, relation, etc.   
 - Can LLM model such visual prio？    

Storyboard image from deviantart.com.   

P113   
## VisorGPT

Can we model such visual prior with LLM

![](./assets/08-113.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.

P114   
## VisorGPT

Prompt design

![](./assets/08-114-1.png) 

![](./assets/08-114-2.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.   

P118   
## VisorGPT

Modeling Visual Prior via Generative Pre-Training

![](./assets/08-118.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.    

P119  
## VisorGPT

Sample from the LLM which has learned visual prior

![](./assets/08-119.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.    

P120   
## VisorGPT

Sample from the LLM which has learned visual prior

![](./assets/08-120.png) 

Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,” NeurIPS 2023.  

P121   
## VideoDirectorGPT

Use storyboard as condition to generate video

![](./assets/08-121.png) 

Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,” arXiv 2023.   


P122  
## VideoDirectorGPT

Use storyboard as condition to generate video

![](./assets/08-122.png) 

Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,” arXiv 2023.    

P124   
## Long-form Video Prior

GPT can be trained to learn better long-form video prior (e.g., object position, relative size, human interaction)

**A new dataset - Storyboard20K**

![](./assets/08-124.png) 

Xie et al., “Learning Long-form Video Prior via Generative Pre-Training,” to be released in 2024.    
<https://github.com/showlab/Long-form-Video-Prior>   


P125  
## Storyboard: More Works
  
|||
|--|--|
|  ![](./assets/08-125-1.png) | **Dysen-VDM** (Fei et al.)<br>Storyboard through scene graphs<br>“Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models,” arXiv 2023. |
| ![](./assets/08-125-2.png)  | **DirectT2V** (Hong et al.) <br> Storyboard through bounding boxes <br> “Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation,” arXiv 2023. |
|  ![](./assets/08-125-3.png)  | **Free-Bloom** (Huang et al.)<br>Storyboard through detailed text prompts<br> “Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator,” NeurIPS 2023. |
|  ![](./assets/08-125-4.png) | **LLM-Grounded Video Diffusion Models** (Lian et al.) <br> Storyboard through foreground bounding boxes <br> “LLM-grounded Video Diffusion Models,” arXiv 2023. |





P126   
## 2 Video Generation

## 2.6 Long video generation


P127  

![](./assets/08-127.png) 


P128  
## NUWA-XL  

Recursive interpolations for generating very long videos

**Method Proposed**

 - A “diffusion over diffusion” architecture for very long video generation

**Key Idea**

 - Key idea: coarse-to-fine hierarchical generation

**Other Highlights**

 - Trained on very long videos (3376 frames)
 - Enables parallel inference
 - Built FlintstonesHD: a new dataset for long video generation, contains 166 episodes with an average of 38000 frames of 1440 × 1080 resolution

Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,” arXiv 2023.   

P129  
## NUWA-XL

Recursive interpolations for generating very long videos

**Generation Pipeline**

 - Storyboard through multiple text prompts   

![](./assets/08-129.png) 

Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,” arXiv 2023. 129

P130  
## NUWA-XL

Recursive interpolations for generating very long videos

**Generation Pipeline**

 - Storyboard through multiple text prompts
 - Global diffusion model: L text prompts → L keyframes
 - Local diffusion model: 2 text prompts + 2 keyframes → L keyframes   

![](./assets/08-130.png) 

Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,” arXiv 2023.


P131  
## NUWA-XL

Recursive interpolations for generating very long videos

**Mask Temporal Diffusion (MTD)**  

 - A basic diffusion model for global & local diffusion models

![](./assets/08-131.png) 


Yin et al., “NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation,” arXiv 2023.    

P133
## Long Video Generation: More Works

|||
|--|--|
| ![](./assets/08-133-1.png)  | **Latent Video Diffusion Models for High-Fidelity Long Video Generation** (He et al.) <br> Generate long videos via autoregressive generation & interpolation <br> “Latent Video Diffusion Models for High-Fidelity Long Video Generation,” arXiv 2022.|
|  ![](./assets/08-133-2.png) | **VidRD** (Gu et al.) <br> Autoregressive long video generation <br> “Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation,” arXiv 2023. |
|  ![](./assets/08-133-3.png) | **VideoGen** (Li et al.) <br> Cascaded pipeline for long video generation <br> “VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation,” arXiv 2023.|

P134  
# 2 Video Generation

## 2.7 Multimodal-guided generation

P135  

![](./assets/08-135.png) 


P136  
## MCDiff  

Motion-guided video generation

![](./assets/08-136.png) 


Chen et al., “Motion-Conditioned Diffusion Model for Controllable Video Synthesis,” arXiv 2023.    

P137  
## MCDiff

Motion-guided video generation

 - Two-stage autoregressive generation

![](./assets/08-137.png) 

Chen et al., “Motion-Conditioned Diffusion Model for Controllable Video Synthesis,” arXiv 2023. 137

P139   
## The Power of Sound (TPoS)

Sound- and text-guided video generation

 - Input/output: a text prompt + an audio segment → a video

![](./assets/08-139.png) 

Jeong et al., “The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion,” ICCV 2023.    

P140
## Sound-Guided Video Generation: More Works

|||
|--|--|
| ![](./assets/08-140-1.png)  | **AADiff** (Lee et al.) <br> “AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion,” CVPRW 2023. |
| ![](./assets/08-140-2.png) | **Generative Disco** (Liu et al.)<br> “Generative Disco: Text-to-Video Generation for Music Visualization,” arXiv 2023. |

Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.   

P141  
## Image-Guided Video Generation: More Works

|||
|--|--|
| ![](./assets/08-141-1.png)  | Generative Image Dynamics (Li et al.) <br> “Generative Image Dynamics,” arXiv 2023. |
|  ![](./assets/08-141-2.png)  | **LaMD** (Hu et al.) <br> “LaMD: Latent Motion Diffusion for Video Generation,” arXiv 2023. |
|  ![](./assets/08-141-3.png)  | **LFDM** (Ni et al.) <br> “Conditional Image-to-Video Generation with Latent Flow Diffusion Models,” CVPR 2023.  |


Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.   


P142  
## Cinematic Mindscapes   

Brain activity-guided video generation

 - Task: human vision reconstruction via fMRI signal-guided video generation   

![](./assets/08-142.png) 

Chen et al., “Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity,” arXiv 2023.     

P144  
## Multimodal-Guided Video Generation: More Works


|||
|--|--|
| ![](./assets/08-144-1.png)  | **MovieFactory** (Zhu et al.) <br> “MovieFactory: Automatic Movie Creation from Text using Large Generative Models for Language and Images,” arXiv 2023. |
| ![](./assets/08-144-2.png) | **CoDi** (Tang et al.) <br> “Any-to-Any Generation via Composable Diffusion,” NeurIPS 2023. |
| ![](./assets/08-144-3.png)  | **MM-Diffusion** (Ruan et al.) <br> “MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation,” CVPR 2023. |
|  ![](./assets/08-144-4.png) | **NExT-GPT** (Wu et al.) <br> “NExT-GPT: Any-to-Any Multimodal LLM,” arXiv 2023.  |
|  |  |




Xing et al., “A Survey on Video Diffusion Models,” arXiv 2023.

P145  
# 3 Video Editing

P146  

![](./assets/08-146.png) 

P147  
# 3 Video Editing

## 3.1 Tuning-based


P148  
## One-Shot Tuned

P149 

![](./assets/08-149.png) 

P150  
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

![](./assets/08-150.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.    

P152   
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

<https://github.com/showlab/Tune-A-Video>

Motivation: appearance from pretrained T2I models, dynamics from a reference video 

![](./assets/08-152.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.   

P153
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

**Obs #1: Still images that accurately represent the verb terms**

![](./assets/08-153-1.png) 

**Obs #2: Extending attention to spatio-temporal yields consistent content**

![](./assets/08-153-2.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023. 153


P154   
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

![](./assets/08-154.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023. 154


P155
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

![](./assets/08-155.png) 

**Full finetuning**: finetunes the entire network

 - inefficient, especially when #frames increases;   
 - prone to overfitting → poor editing ability.   

**Our tuning strategy**: update the specific projection matrices

 - parameter efficient and fast (~10 min);    
 - retains the original property of pre-trained T2I diffusion models.   


**Structure guidance via DDIM inversion**

\begin{align*} \mathcal{V} ^\ast =\mathcal{D} (\mathrm{DDIM-samp} (\mathrm{DDIM-inv} (\varepsilon (\mathcal{V} )),\tau^\ast  ))\end{align*}

 - preserves the structural information   
 - improves temporal consistency   



Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023. 155


P156  
## Tune-A-Video

![](./assets/08-156.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.    

P157  
## Tune-A-Video

![](./assets/08-157.png) 


Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.

P158  
## Tune-A-Video

![](./assets/08-158.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.   

P159  
## Tune-A-Video

![](./assets/08-159.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.

P160  
## Tune-A-Video


![](./assets/08-160.png) 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.

P161
## Tune-A-Video

One-shot tuning of T2I models for T2V generation/editing

![](./assets/08-161.png) 

**Automatic metrics – CLIP Score**

 - *Frame Consistency*: the average cosine similarity between all pairs of video frames   
 - *Textual Alignment*: average CLIP score between all frames of output videos and corresponding edited prompts    

**User study** 

Compare two videos generated by our method and a baseline (shown in random order):   

 - *Which video has better temporal consistency?*    
 - *Which video better aligns with the textual description?*    

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.

P162  
## Dreamix

Few-shot finetuning for personalized video editing

**Main idea: Mixed Video-Image Finetuning**

 - Finetune Imagen Video (Ho et al., 2022) which is a strong video foundation model

![](./assets/08-162-1.png) 

 - Finetuned to generate individual frames (bypassing temporal attentions) & video

![](./assets/08-162-2.png) 

Molad et al, “Dreamix: Video Diffusion Models are General Video Editors,” arXiv 2023. 162


P163  
## Dreamix

Few-shot finetuning for personalized video editing

**Inference Overview**

 - Corrupt the input video by downsampling and add noise   
 - Apply the finetuned video diffusion model to denoise and upscale   

![](./assets/08-163.png) 

Molad et al, “Dreamix: Video Diffusion Models are General Video Editors,” arXiv 2023.    

P164   
## One-Shot Tuned Video Editing: More Works

|||
|--|--|
| ![](./assets/08-164-1.png)  | \\(\mathbf{EI^2}\\) (Zhang et al.)<br> Modify self-attention for better temporal consistency <br> “Towards Consistent Video Editing with Text-to-Image rDiffusion Models,” arXiv 2023. |
| ![](./assets/08-164-2.png)  |  **Video-P2P** (Liu et al.) <br> Improve input-output semantic consistency of video editing via shared embedding optimization and cross-attention control <br> “Video-P2P: Video Editing with Cross-attention Control,” arXiv 2023. |

P165   
## One-Shot Tuned Video Editing: More Works

![](./assets/08-165.png) 

**Compared to training-free editing methods:** 

 - Cons: still need 1 video for training
 - Pros: supports significant shape change 

Wu et al., “Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation,” ICCV 2023.    

P166  
## Multiple-Shot Tuned

Video Editing: Text Conditioned

P167  

![](./assets/08-167.png) 

P168
## MotionDirector

Tune on multiple videos of a motion to be customised



![](./assets/08-168-1.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.    

P169  
## MotionDirector

Tune on multiple videos of a motion to be customised

![](./assets/08-169.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.    

P170   
## MotionDirector

Tune on multiple videos of a motion to be customised

![](./assets/08-170.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023   

P171   
## MotionDirector

## Tune on multiple videos of a motion to be customised

 - MokonDirector can customize foundakon models to generate videos with desired mokons.

![](./assets/08-171.png) 


Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.   

P172   
## MotionDirector

Tune on multiple videos of a motion to be customised  

 - The challenge is generalizing the learned motions to diverse appearance.   
 - MotionDirector learns the appearances and motions in reference videos in a decoupled way, to avoid overfitting on the limited appearances.   


![](./assets/08-172.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.   

P173   
## MotionDirector

Tune on multiple videos of a motion to be customised

 - Decouple appearance and motion.

![](./assets/08-173.png) 

Zhao et al., “MogonDirector: Mogon Customizagon of Text-to-Video Diffusion Models,” arXiv 2023.     

P174   
## MotionDirector  


Tune on muleple videos of a moeon to be customised

- Comparing with other methods.

![](./assets/08-174.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.    

P175   
## MotionDirector

Tune on multiple videos of a motion to be customised

 - Comparing with other methods.   

![](./assets/08-175.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.    

P176  
## MofonDirector

Tune on multiple videos of a motion to be customised

![](./assets/08-176.png) 

Zhao et al., “MotionDirector: Motion Customization of Text-to-Video Diffusion Models,” arXiv 2023.     

P177  
# 3 Video Editing

## 3.2 Training-free

P178   
![](./assets/08-178.png) 

P179   
## TokenFlow

Consistent high-quality semantic edits

Main challenge using T2I to edit videos without finetuning: temporal consistency  

![](./assets/08-179.png) 

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Edigng,” arXiv 2023.    

P180   
## TokenFlow

Consistent high-quality semantic edits

**Key Idea**

 - Achieve consistency by enforcing the inter-frame correspondences in the original video   

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Edigng,” arXiv 2023.     

P181   
## TokenFlow   

Consistent high-quality semanec edits

**Main idea**

![](./assets/08-181.png) 

Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.    

P182   
## TokenFlow

Consistent high-quality semantic edits

**Main idea**

During conditional denoising, use features from corresponding positions in preceding and following frames instead of the pixel's own feature at output of extended-attention

|||
|--|--|
| ![](./assets/08-182.png)  | ![](./assets/08-182-1.png)  |


Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.    

P183   
## TokenFlow

Consistent high-quality semantic edits

![](./assets/08-183.png) 

No temporal finetuning, good temporal consistency Geyer et al., “TokenFlow: Consistent Diffusion Features for Consistent Video Editing,” arXiv 2023.     

P184   
## FateZero   

Attention map fusing for better temporal consistency

**Methodology** 

 - During DDIM inversion, save inverted self-/cross-attention maps    
 - During editing, use some algorithms to blend inverted maps and generated maps   

![](./assets/08-184.png) 

Qi et al., “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing,” ICCV 2023.    

P184   
## FateZero   

Attention map fusing for better temporal consistency

**Methodology**

 - During DDIM inversion, save inverted self-/cross-avenkon maps   
 - During edikng, use some algorithms to blend inverted maps and generated maps   

![](./assets/08-185.png) 

Qi et al., “FateZero: Fusing Attentions for Zero-shot Text-based Video Editing,” ICCV 2023.    


P186    
## FateZero  

Attention map fusing for better temporal consistency

![](./assets/08-186.png) 

Qi et al., “FateZero: Fusing Akengons for Zero-shot Text-based Video Edigng,” ICCV 2023.    

P187   
## Training-Free Video Editing: More Works


|||
|--|--|
| ![](./assets/08-187-1.png)  | **MeDM** (Chu et al.) <br> OpScal flow-based guidance for temporal consistency <br> “MeDM: Mediagng Image Diffusion Models for Video-to￾Video Translagon with Temporal Correspondence Guidance,” arXiv 2023. |
| ![](./assets/08-187-2.png) | **Ground-A-Video** (Jeong et al.) <br> Improve temporal consistency via modified attention and optical flow <br> “Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models,” arXiv 2023. |
| ![](./assets/08-187-3.png)  | **Gen-L-Video** (Lorem et al.) <br> Edit very long videos using existing generators <br> “Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising,” arXiv 2023.  |
| ![](./assets/08-187-4.png)  | **FLATTEN** (Cong et al.) <br> Optical flow-guided attention for temporal consistency <br> “Flatten: optical flow-guided attention for consistent text-to-video editing,” arXiv 2023. |
| ![](./assets/08-187-5.png) | **InFusion** (Khandelwal et al.) <br> Improve temporal consistency via fusing latents <br> “InFusion: Inject and Attention Fusion for Multi Concept Zero-Shot Text-based Video Editing,” ICCVW 2023.  |
| ![](./assets/08-187-6.png)  | **Vid2Vid-Zero** (Wang et al.) <br> Improve temporal consistency via cross￾attention guidance and null-text inversion <br> “Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models,” arXiv 2023. |


P188    
# 3 Video Editing

## 3.3 Controlled Edifng (depth/pose/point/ControlNet)

P189   

![](./assets/08-189.png) 

P190   
## Depth Control


P192   
## Use MiDaS to offer depth condition

Depth estimating network


Ranftl et al., “Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer,” TPAMI 2022.


P193   
## Gen-1

Framewise depth-guided video editing

 - Inflate Stable Diffusion to a 3D model, finetune on pretrained weights   
 - Insert temporal convolution/attention layers   
 - Finetune to take **per-frame depth as conditions**   

|||
|--|--|
| ![](./assets/08-193-1.png)  | ![](./assets/08-193-2.png)  |


Psser et al., “Structure and Content-Guided Video Synthesis with Diffusion Models,” ICCV 2023. 

P194   
## Pix2Video

Framewise depth-guided video editing

 - Leverage a pretrained per-frame depth-conditioned Stable Diffusion model to edit frame by frame, to maintain motion consistency between source video and edited video
 - No need for training/finetuning
 - Challenge is how to ensure temporal consistency?   

![](./assets/08-194.png) 

Ceylan et al., “Pix2Video: Video Editing using Image Diffusion,” ICCV 2023.   


P195   
## Pix2Video

Framewise depth-guided video editing

 - How to ensure temporal consistency?   
    - Obtain initial noise from DDIM inversion   

![](./assets/08-195.png) 

Ceylan et al., “Pix2Video: Video Editing using Image Diffusion,” ICCV 2023.    

P196   
## Pix2Video

Framewise depth-guided video editing

 - How to ensure temporal consistency?    
    - Inject self-attention features from the previous frame in U-Net for generating the current frame    
    - Use the latent of the previous frame to guide latent update of the current frame   
 
![](./assets/08-196.png) 

Ceylan et al., “Pix2Video: Video Editing using Image Diffusion,” ICCV 2023.    

P197   
## Pix2Video

Framewise depth-guided video editing

![](./assets/08-197.png) 

Ceylan et al., “Pix2Video: Video Editing using Image Diffusion,” ICCV 2023   

P198   
## Pix2Video

Framewise depth-guided video editing

![](./assets/08-198.png) 

Ceylan et al., “Pix2Video: Video Editing using Image Diffusion,” ICCV 2023.


P199   
## ControlNet / Multiple Control

P200   
## ControlVideo (Zhang et al. 2023)

ControlNet-like video editing

 - Input structural conditions through **ControlNet**

![](./assets/08-200.png) 


Zhang et al., “ControlVideo: Training-free Controllable Text-to-Video Generation,” arXiv 2023. 200

P201   
## ControlVideo (Zhang et al. 2023)

ControlNet-like video editing

 - Use pretrained weights for Stable Diffusion & ControlNet, no training/finetuning   
 - Inflate Stable Diffusion and ControlNet along the temporal dimension   
 - Interleaved-frame smoothing during DDIM sampling for bever temporal consistency    

![](./assets/08-201.png) 


Zhang et al., “ControlVideo: Training-free Controllable Text-to-Video Generation,” arXiv 2023. 201

P202   
## ControlVideo (Zhang et al. 2023)  

ControlNet-like video editing

 - Use pretrained weights for Stable Diffusion & ControlNet, no training/finetuning   
 - Inflate Stable Diffusion and ControlNet along the temporal dimension    
 - Interleaved-frame smoothing during denoising for better temporal consistency    

![](./assets/08-202.png)

Zhang et al., “ControlVideo: Training-free Controllable Text-to-Video Generation,” arXiv 2023.    

P203   
## ControlVideo (Zhang et al. 2023)   

ControlNet-like video editing

![](./assets/08-203.png)  

Zhang et al., “ControlVideo: Training-free Controllable Text-to-Video Generation,” arXiv 2023.     

P208   
## VideoControlNet

Optical flow-guided video editing; I, P, B frames in video compression

![](./assets/08-208.png)  

Hu et al., “VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet,” arXiv 2023.     


P210   
## CCEdit

Mulemodal-guided video edieng

![](./assets/08-210.png)  

Feng et al., “CCEdit: Creative and Controllable Video Editing via Diffusion Models,” arXiv 2023.    

P211   
## VideoComposer

Image-, sketch-, motion-, depth-, mask-controlled video editing

**Video Editing based on Various Conditions**

![](./assets/08-211.png)  

Wang et al., “VideoComposer: Compositional Video Synthesis with Motion Controllability,” arXiv 2023.  

P212   
## VideoComposer

Image-, sketch-, motion-, depth-, mask-controlled video editing   

• Spako-Temporal Condikon encoder (STC-encoder): a unified input interface for condikons   

![](./assets/08-212.png)  

Wang et al., “VideoComposer: Compositional Video Synthesis with Motion Controllability,” arXiv 2023.    

P214   
## ControlNet- and Depth-Controlled Video Editing: More Works

|||
|--|--|
| ![](./assets/08-214-1.png)  | **MagicProp** (Yan et al.) <br> “MagicProp: Diffusion-based Video Editing via Motion-aware Appearance Propagation,” arXiv 2023. |
| ![](./assets/08-214-2.png) | **Make-Your-Video** (Xing et al.) <br> “Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance,” arXiv 2023.   |
| ![](./assets/08-214-3.png) | **Control-A-Video** (Lorem et al.) <br> “Control-A-Video: Controllable Text-to-Video Generagon with Diffusion Models,” arXiv 2023. |
| ![](./assets/08-214-4.png)  | **MagicEdit** (Liew et al.) <br> “MagicEdit: High-Fidelity and Temporally Coherent Video Editing,” arXiv 2023. |
| ![](./assets/08-214-5.png)  |  **EVE** (Chen et al.) <br> “EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints,” arXiv 2023. |

P215   
## Pose Control

P216  
## DreamPose

Pose- and image-guided video generation

Input: image  \\(\quad \\) Input: pose sequence   \\(\quad \\)  Output: Video   

 
Karras et al., “DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion,” arXiv 2023.   

P217   
## MagicAnimate

Pose- and image-guided video generation

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animagon using Diffusion Model,” arXiv 2023.    

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

P219   
## MagicAnimate

Pose- and image-guided video generation   

• Training data: TikTok dataset of 350 dancing videos; TED-talks dataset of 1,203 video clips   

![](./assets/08-219.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023. 219

P220   
## MagicAnimate

Pose- and image-guided video generation

![](./assets/08-220.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.



P223   
## MagicAnimate

Pose-guided video generation

![](./assets/08-223.png) 

Xu et al., “MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model,” arXiv 2023.    

P224   
## Video Editing Under Pose Guidance: More Works

|||
|--|--|
| ![](./assets/08-224-1.png)  | **Dancing Avatar** (Qin et al.)<br> Pose-guided video editing <br> “Dancing avatar: Pose and text-guided human motion videos synthesis with image diffusion model,” arXiv 2023. |
| ![](./assets/08-224-2.png)  | **Follow Your Pose** (Ma et al.) <br> Pose-guided video editing  <br> “Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos,” arXiv 2023.  |
| ![](./assets/08-224-3.png)  | **DisCo** (Wang et al.) <br> Pose-guided video editing <br> “Disco: Disentangled control for referring human dance generation in real world,” arXiv 2023.  |

P225   
## Point-Control

P226   
## VideoSwap

Customized video subject swapping via point control

**Problem Formulation**

 - Subject replacement: change video subject to a **customized** subject    
 - Background preservation: preserve the unedited background same as the source video    

![](./assets/08-226.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P227    
## VideoSwap

Customized video subject swapping via point control


**Motivation**

 - Existing methods are promising but still often motion not well aligned   
 - Need ensure precise correspondence of <u> **semantic points** </u> between the source and target   

![](./assets/08-227.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P228    
## VideoSwap

Customized video subject swapping via point control

**Empirical Observations**

 - **Question**: Can we <u> learn semantic point control </u> for a specific <u>source video subject</u> using only a <u>small number of source video frames</u>   
 - **Toy Experiment**: Manually define and annotate a set of semantic points on 8 frame; use such point maps as condition for training a control net, i.e., T2I-Adapter.    

![](./assets/08-228.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.

P229    
## VideoSwap

Customized video subject swapping via point control

**Empirical Observations**

 - **Observation 1**: If we can drag the points, the trained T2I-Aapter can generate new contents based on such dragged new points (new condition) à feasible to use semantic points as condition to control and maintain the source motion trajectory.

![](./assets/08-229.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.

P230    
## VideoSwap

Customized video subject swapping via point control

**Empirical Observations**

 - **Observation 2**: Further, we can drag the semantic points to control the subject’s shape   
 
![](./assets/08-230.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   

P231    
## VideoSwap

Customized video subject swapping via point control

![](./assets/08-231.png) 

**Framework**

 - **Motion layer**: use pretrained and fixed AnimateDiff to ensure essential temporal consistency    
 - **ED-LoRA** \\(_(Mix-of-Show)\\): learn the wconcept to be customized   

 - **Key design aims**: 
    - Introduce semantic point correspondences to guide motion trajectory   
    - Reduce human efforts of annotating points    


Gu et al. “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   
Gu et al. “Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models.” NeurIPS, 2023.   

P232   
## VideoSwap

Customized video subject swapping via point control

**Step 1: Semantic Point Extraction**

 - Reduce human efforts in annotating points    
    - User define point at one keyframe    
    - Propagate to other frames by point tracking/detector   
 - Embedding    

![](./assets/08-232.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   

P233   
## VideoSwap

Customized video subject swapping via point control

**Methodology – Step 1: Semantic Point Extraction on the source video**


 - Reduce human efforts in annotating points   
 - Embedding   
    - Extract DIFT embedding (intermediate U-Net feature) for each semantic point   
    - Aggregate over all frames   

![](./assets/08-233.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   

P234    
## VideoSwap

Customized video subject swapping via point control

**Methodology – Step 2: Semantic Point Registration on the source video**  

 - Introduce several learnable MLPs, corresponding to different scales
 - Optimize the MLPs    
    - Point Patch Loss: restrict diffusion loss to reconstruct local patch around the point    
    - Semantic-Enhanced Schedule: only sample higher timestep (0.5T, T), which prevents overfitting to low-level details    

![](./assets/08-234.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P235    
## VideoSwap

Customized video subject swapping via point control   

**Methodology**   

 - After Step1 (Semantic Point Extraction) and Step2 (Semantic Point Registration), those semantic points can be used to guide motion   
 - User-point interaction for various applications   

![](./assets/08-235.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P236   
## VideoSwap

Customized video subject swapping via point control

**Methodology**

 - How to drag point for shape change?   
    - Dragging at one frame is straightforward, propagating drag displacement over time is non-trivial, because of complex camera motion and subject motion in video.   
    - Resort to canonical space (i.e., Layered Neural Atlas) to propagate displacement.   

![](./assets/08-236.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P237   
## VideoSwap

Customized video subject swapping via point control

**Methodology**

 - How to drag point for shape change?   
 - Dragging at one frame is straightforward, propagating drag displacement over time is non-trivial because of complex camera motion and subject motion in video.   
 - Resort to canonical space (i.e., Layered Neural Atlas) to propagate displacement.    

![](./assets/08-237.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P238    
## VideoSwap

Customized video subject swapping via point control

![](./assets/08-238-1.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   

P239   
## VideoSwap

Customized video subject swapping via point control

![](./assets/08-239.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.   

P240   
## VideoSwap

Customized video subject swapping via point control

**Qualitative Comparisons to previous works**

 - VideoSwap can **support shape change** in the target swap results, leading to the correct identity of target concept. 

![](./assets/08-240.png) 

Gu et al., “VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence,” 2023.    

P242   
# 3 Video Editing

## 3.4 3D-Aware

P243   

![](./assets/08-243.png) 


P244   
## Layered Neural Atlases

Decompose a video into two images

 - Decompose a video into a foreground image + a background image    

![](./assets/08-244.png) 

Kasten et al., “Layered Neural Atlases for Consistent Video Editing,” arXiv 2023.   

P245    
## Layered Neural Atlases

Decompose a video into two images

 - Decompose a video into a foreground image + a background image   
 - Edit the foreground/background image = edit the video    

![](./assets/08-245.png) 

Kasten et al., “Layered Neural Atlases for Consistent Video Editing,” arXiv 2023.    


P246   
## VidEdit

Atlas-based video editing

 - Decompose a video into a foreground image + a background image   
 - Edit the foreground/background image = edit the video   
 - Use diffusion to edit foreground/background atlas   

![](./assets/08-246.png) 

Video from Kasten et al., “Layered Neural Atlases for Consistent Video Edigng,” arXiv 2023.    
Couairon et al., “VidEdit: Zero-Shot and Spagally Aware Text-Driven Video Edigng,” arXiv 2023.   

P247    
## StableVideo & Shape-aware Text-drive Layered Video Editing

Atlas-based video edieng

![](./assets/08-247.png) 

Lee et al., “Shape-aware Text-driven Layered Video Editing,” CVPR 2023.   
Chai et al., “StableVideo: Text-driven Consistency-aware Diffusion Video Editing,” ICCV 2023.   

P248   
## StableVideo

Atlas-based video editing


![](./assets/08-248.png) 

Chai et al., “StableVideo: Text-driven Consistency-aware Diffusion Video Edigng,” ICCV 2023.   

P249    
## Content Deformation Field (CoDeF)

Edit a video = edit a canonical image + learned deformaeon field

 - Limitations of Neural Layered Atlases   
    - Limited capacity for faithfully reconstructing intricate video details, missing subtle motion features like blinking eyes and slight smiles   
    - Distorted nature of the estimated atlas leads to impaired semantic information   

 - Content Deformation Field: inspired by dynamic NeRF works, a new way of representing video, as a 2d canonical image + 3D deformation field over time   

 - Edit a video = edit a canonical image + learned deformation field   


Ouyang et al., “CoDeF: Content Deformation Fields for Temporally Consistent Video Processing,” arXiv 2023.


P250    
## Content Deformation Field (CoDeF)

Edit a video = edit a canonical image + learned deformation field    

**Problem Formulation**

 - Decode a video into a 2D canonical field and a 3D temporal deformation field   
 - Deformation Field: video (x, y, t) → canonical image coordinate (x’, y’)   
 - Canonical Field: (x’, y’) → (r, g, b), like a “2D image”   

![](./assets/08-250.png) 

Ouyang et al., “CoDeF: Content Deformation Fields for Temporally Consistent Video Processing,” arXiv 2023.   

P251   
## Content Deformation Field (CoDeF)

Edit a video = edit a canonical image + learned deformation field    

**CoDeF compared to Atlas**

 - Superior robustness to non-rigid motion   
 - Effective reconstruction of subtle movements (e.g. eyes blinking)    
 - More accurate reconstruction: 4.4dB higher PSNR   

Ouyang et al., “CoDeF: Content Deformation Fields for emporally Consistent Video Processing,” arXiv 2023.   

P252   
## Content Deformation Field (CoDeF)

Edit a video = edit a canonical image + learned deformation field   

![](./assets/08-252.png) 

Ouyang et al., “CoDeF: Content Deformation Fields for emporally Consistent Video Processing,” arXiv 2023.   

P253   
## Content Deformafon Field (CoDeF)

Edit a video = edit a canonical image + learned deformation field   

![](./assets/08-253.png) 

Ouyang et al., “CoDeF: Content Deformation Fields for emporally Consistent Video Processing,” arXiv 2023.   

P253   
## DynVideo-E

Edit a video = edit a canonical ~~image~~ 3D NeRF   

Canonical image in CoDeF is still 2D   

Can we represent the video in a truly 3D space?   

P255   
## DynVideo-E

Edit a video = edit a canonical ~~image~~ 3D NeRF

![](./assets/08-255.png) 

Liu et al., “DynVideo-E: Harnessing Dynamic NeRF for arge-Scale Motion- and View-Change Human-Centric Video Editing,” arXiv 2023.   

P257   
## DynVideo-E

Edit a video = edit a canonical ~~image~~ 3D NeRF   

**Main idea**

 - For the first time introduce the dynamic NeRF as an innovative video representation for large-scale motion- and view-change human-centric video editing.   

Liu et al., “DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing,” arXiv 2023.

P258  
## DynVideo-E

Edit a video = edit a canonical ~~image~~ 3D NeRF   

Follow HOSNeRF, represent the video as:   
 - Background NeRF   
 - Human NeRF   
 - Deformation Field   

Edit background NeRF and human NeRF respectively   

![](./assets/08-258.png) 

Liu et al., “HOSNeRF: Dynamic Human-Object-Scene Neural Radiance Fields from a Single Video,” ICCV 2023.   
Liu et al., “DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Mogon- and View-Change Human-Centric Video Edigng,” arXiv 2023.   


P259  
## DynVideo-E

DynVideo-E significantly outperforms SOTA    
approaches on two challenging datasets by a large margin of 50% ∼ 95% in terms of human preference   

P263   
# 3 Video Editing  

## 3.5 Other Guidance


P264  

![](./assets/08-264.png) 

P265   
## InstructPix2Pix

Instruction-guided image editing

![](./assets/08-265.png) 

Brooks et al., “InstructPix2Pix: Learning to Follow Image diting Instructions,” CVPR 2023.  


P266   
## InstructVid2Vid

Instruction-guided Video Editing

 - Generate ⟨instruction, video⟩ dataset using ChatGPT, BLIP and Tune-A-Video   
 - Train inflated Stable Diffusion for instruction-guided video editing   

![](./assets/08-266.png) 

Qin et al., “InstructVid2Vid: Controllable Video Editing with Natural Language Instructions,” arXiv 2023.  

P267   
## Speech Driven Video Editing via an Audio-Conditioned Diffusion Model

Speech-driven video editing

![](./assets/08-267.png) 

Bigioi et al., “Speech Driven Video Editing via an Audio-Conditioned Diffusion Model,” arXiv 2023.   

P268   
## Soundini

Sound-guided video editing

![](./assets/08-268.png) 

Lee et al., “Soundini: Sound-Guided Diffusion for Natural Video Editing,” arXiv 2023.    

P269   
## Video Editing Under Various Guidance: More Works

|||
|--|--|
| ![](./assets/08-269-1.png)  | **Collaborative Score Distillation** (Kim et al.) <br> Instruction-guide video editing <br> “Collaborative Score Distillation for Consistent Visual Synthesis,” NeurIPS 2023. |
| ![](./assets/08-269-2.png)  | **Make-A-Protagonist** (Zhao et al.) <br> Video ediSng with an ensemble of experts <br> “Make-A-Protagonist: Generic Video Edigng with An Ensemble of Experts,” arXiv 2023. |
| ![](./assets/08-269-3.png)  | **DragNUWA** (Yin et al.) <br> Multimodal-guided video editing <br> “DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory,” arXiv 2023. |



