# Diffusion model architectures
P4   
## Architecture

P5   
### U-Net Based Diffusion Architecture

#### U-Net Architecture

![](../assets/D2-5-1.png) 

> &#x2705; U-Net的是Large Scale Image Diffusion Model中最常用的backbone。  

> &#x1F50E; Ronneberger et al., <u>“U-Net: Convolutional Networks for Biomedical Image Segmentation”, </u>MICCAI 2015    

#### Pipeline 

![](../assets/D2-5-2.png) 

> &#x2705; 包含Input、U-Net backbone、Condition。  
> &#x2705; Condition 通常用 Concat 或 Cross attention 的方式与 Content 相结合。    

> &#x1F50E; Rombach et al., <u>"High-Resolution Image Synthesis with Latent Diffusion Models",</u> CVPR 2022    

U-Net Based Diffusion Architecture

P6   
#### Application

**Imagen** Saharia et al.   

Saharia et al. <u>“Photorealistic text-to-image diffusion models with deep language understanding”, </u>NeurIPS 2022    

**Stable Diffusion** Rombach et al.    

Rombach et al., <u>"High-Resolution Image Synthesis with Latent Diffusion Models", </u>CVPR 2022    

**eDiff-I** Balaji et al.    

Balaji et al.,” <u>ediffi: Text-to-image diffusion models with an ensemble of expert denoisers”, </u>arXiv 2022    

P7    
### Transformer Architecture

#### Vision Transformer(ViT)

![](../assets/D2-7-1.png) 

Dosovitskiy et al., <u>“An image is worth 16x16 words: Transformers for image recognition at scale”, </u>ICLR 2021    

#### Pipeline

![](../assets/D2-7-2.png) 

> &#x2705; 特点：  
> &#x2705; 1. 把 image patches 当作 token.    
> &#x2705; 2. 在 Shallow layer 与 deep layer 之间引入 long skip connection.    

Bao et al.,<u> "All are Worth Words: a ViT Backbone for Score-based Diffusion Models", </u>arXiv 2022    


P8   
#### Application

Peebles and Xie, <u>"Scalable Diffusion Models with Transformers", </u>arXiv 2022    
Bao et al., <u>"One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale", </u>arXiv 2023    
Hoogeboom et al., <u>"simple diffusion: End-to-end diffusion for high resolution images", </u>arXiv 2023    


P9    
# Image editing and customization with diffusion models

P10    

## SDEdit

Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", </u>ICLR 2022 

### guided synthesis／editing Task   

![](../assets/D2-11.png) 

> &#x2705; 过去的 guided synthesis／editing 任务是用 GAN based 方法实现的。    

P12   
### Pipeline

![](../assets/D2-12.png) 

**Gradually projects the input to the manifold of natural images.**

> 准备工作：一个预训练好的Image Diffusion Model  
> 第一步：perturb the input with **Gaussian noise**  
> 第二步：progressively remove the noise using a pretrained diffusion model.    
> [?] 怎样保证Denoise的图像与原始图像有高度的一致性？

### 特点

> 只需要一个预训练模型，不需要额外的finetune。  

P13   
### 其它应用场景

#### Fine-grained control using strokes

![](../assets/D2-13.png) 

> 可以在Image上加上草图，也可以直接使用草图生成图像  

P16   
#### Image compositing  

![](../assets/D2-16.png) 

> 把上面图的指定pixel patches应用到下面图上。  
> SDEdit的结果更合理且与原图更像。  

P17   
### 效率提升

Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

![](../assets/D2-17.png) 

原理：全图生成速度比较慢，因此针对被编辑区域进行部分生成。  

Li et al., <u>"Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models", </u>NeurIPS 2022    

P18   

## Style Transfer with DDIM inversion

### Recap DDIM Inversion

![](../assets/D2-18.png) 

Song et al., <u>"Denoising Diffusion Implicit Models",</u> ICLR 2021    

> &#x2705; DDIM 方法中，从 noise 到图像的映射关系是确定的。同样也可以让图像与 noise 的关系是确定的。这个过程称为 DDIM Inversion.    
> &#x2705; DDIM Inversion 是图像编辑的常用方法。     

P19   
### Pipeline

![](../assets/D2-19.png) 

Su et al., <u>"Dual diffusion implicit bridges for image-to-image translation", </u>ICLR 2023    

> &#x2705; 假设已有一个 文生图的pretrained DDIM model．    
> &#x2705; 任务：把老虎的图像变成猫的图像，且不改变 Sryle.     
> &#x2705; (1) 老虎图像 ＋ DDIM Inversion ＋ “老虎”标签  → noise      
> &#x2705; (2) noise ＋ DDIM ＋ “猫”标签 → 猫图像        
> &#x2705; 优点：不需要重训。     


P20   
### 效果

![](../assets/D2-20.png) 

Su et al., <u>"Dual diffusion implicit bridges for image-to-image translation",</u> ICLR 2023    


> &#x2705; 学习不同颜色空间的 transform    


P21    
## DiffEdit: Diffusion-based semantic image editing with mask guidance    

Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance", </u>ICLR 2023    

### 任务目标

> SDEdit要求用户对想更新的区域打MASK。  

Instead of asking users to provide the mask, the model will generate the mask itself based on the caption and query.    


![](../assets/D2-21.png) 


> &#x2705; 作者认为让用户打 MASK 比较麻烦，因此生成 MASK    


P22   
### Pipeline

![](../assets/D2-22.png)  

P23   
### 效果
![](../assets/D2-23.png)      

> 生成质量高且与原始相似度高。  

P24   
## Imagic: Text-Based Real Image Editing with Diffusion Models   

![](../assets/D2-24.png) 

Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models",</u> CVPR 2023     

> &#x2705; 对内容进行复杂修改，但对不相关部分保留。    


P25   
### Pipeline

> 输入：Origin Image和target text promt

![](../assets/D2-25-1.png)     

> &#x2705; Step 1： 对 target text 作 embedding，得到init text embedding \\(e_{tgt}\\)。然后优化init text embedding，使得Pre-Trained Diffusion Model可以根据Optimized text embedding \\(e_{opt}\\) 重建出Origin Image。
 
![](../assets/D2-25-2.png)     

> &#x2705; Step 2： 用 Optimized text embedding \\(e_{opt}\\) 重建 Origin Image，这一步会 finetune diffusion model。   

![](../assets/D2-25-3.png)    

> &#x2705; Step 3：用finetuned diffusion model生成target Image。其中condition为\\(e_{tgt}\\)和\\(e_{opt}\\)的插值。  


P26   
### 效果

![](../assets/D2-26.png) 


P27   
## Prompt-to-Prompt Image Editing with Cross-Attention Control

![](../assets/D2-27.png)

> &#x2705; 基于标题的图像编辑 (1) 修改某个单词的影响力；(2) 替换单词；(3) 添加单词；     

Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control", </u>ICLR 2023    

P28   
### Pipeline

![](../assets/D2-28.png)

> &#x2705; 控制生成过程中的 attention maps。其具体方法为，在每个step中，把原始图像的 attention map 注入到 diffusion 过程中。  
> 图中上面步骤描述正常的文生图的cross attention设计。  
> 图中下面步骤描述了如何控制cross attention过程中的attention map。三种控制方式分别对应三种图像编辑方法。      


P29   
### 效果

![](../assets/D2-29.png)    

P30   
## InstructPix2Pix: Learning to Follow Image Editing Instructions  

![](../assets/D2-30.png)   

Brooks et al., <u>"Instructpix2pix: Learning to follow image editing instructions”,</u> CVPR 2023    

> 控制特点：不需要完整的控制文本，只需要告诉模型要怎么修改图像。  
> &#x2705; 优势：只修改推断过程，不需针对图像做 finetune.    

P31   

### Pipeline
![](../assets/D2-31-1.png)    

> 生成Image Editing Dataset：  
Step a：微调GPT-3，用于生成Instruction和Edited Caption。  
Step b：使用预训练模型生成pair data。  

![](../assets/D2-31-2.png)    

> &#x2705; [?]只是文本引导方式做了改变，哪里体现 pix 2 呢？     

P32   
## Personalization with diffusion models   

![](../assets/D2-32.png) 

> &#x2705; 基于目标的多张 reference，输入文本，生成包含目标的图像。要求生成的结果与refernce一致，且具有高质量和多样性：     

![](../assets/D2-33.png) 

Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    

P34   
### Pipeline  

![](../assets/D2-34.png)     
 
> &#x2705; 使用 reference image 微调 model，具体方法为：    
> &#x2705; 输入多张reference image，使用包含特定 identifier 的文本构造 pairdata。目的是对输入图像做 encode。    
> &#x2705; 同时使用用不含 identifer 的图像和文本调练，构造重建 loss 和对抗 loss.目的是生成的多样性及防止过拟合。    

P35   
### DreamBooth Results

![](../assets/D2-35.png)  

> Input Image的基本特征保持住了，但是细节还是有些丢失。比如书包右下角的三个贴图，在每个生成里面都不一样。  

P36   
### DreamBooth Applications    

![](../assets/D2-36.png) 

> 用来生成动作照片还是可以的，因为人对动画的细节差异没有那么敏感。例如这只猫。额头上的花纹，在每张图像上都不一样。如果用来生成人，会发明显的差异。  

P37    
## Textual Inversion: Optimizing Text Embedding   

![](../assets/D2-37.png) 

> &#x2705; 输入3-5张reference iamge。可以把内容、风格、动作等编辑为 \\(S_ {\ast }\\)     
> &#x2705; 用一个 word 来 Encode 源，因此称为 Textual Inversion.    

Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023    

P38   
### Pipeline

![](../assets/D2-38.png) 

> &#x2705; 输入带 holder world 的 String，(1) 转为 token (2) token 转为“连续的表示”，即 embedding. (3) embedding 转为 conditional code，用于引导生成模型。    
> &#x2705; 通过生成的结果与GT比较，构造重建loss来优化 embedding.    


P39   
### Textual Inversion Results   

![](../assets/D2-39.png) 

P40    
Works well for artistic styles    

![](../assets/D2-40.png) 

P41   
## Low-rank Adaptation (LoRA)   

 - Lora: Low-rank adaptation of large language models

> &#x2705; 要解决的问题：finetune 所需的训练时间、参数存储，Computation 的成本很高。    

![](../assets/D2-41.png) 

> &#x2705; 解决方法：仅仅拟合residual model 而不是 finetune entire model.  
> 残差通常可以用low rank Matrix来拟合，因为称为low-rank adaptation。    


Lora [Edward J. Hu\\(^\ast \\), Yelong Shen\\(^\ast \\), et al., ICLR 2022]     
Lora + Dreambooth (by Simo Ryu): <https://github.com/cloneofsimo/lora>     

### LoRA Results

> &#x2705; LoRA对数据集要求少，收敛速度快。可以极大提升 finetune 效率，也更省空间。    

P43
## Multi-Concept Customization of Text-to-Image Diffusion

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

### finetune的性能问题

> 要解决的问题与上一篇相同，即finetune 所需的训练时间、参数存储，Computation 的成本很高。
> 但解决方法有些区别。上一篇通过增加额外的残差模块，而这一篇通过只finetune原始模型的部分参数。  

P45   
#### Analyze change in weights   

> &#x2705; 选择模型的部分参数进行 finetune．问题是怎么选择？    
> 作者通过分析模型各参数的重要性，insights 应该 finetune 哪些参数。   

![](../assets/D2-45.png) 

> &#x2705; Cross-Attn 层用于结合图像和文本的特征。     
> &#x2705; Self-Attn 用于图像内部。    
> &#x2705; Other 主要是卷积和 Normalization.    
> &#x2705; 通过比较 pretrained 模型和 finetune 模型，change 主要发生成Cross-Attn 层，说明 Cross-Attn 层在 finetune 过程中更重要！    

P46   
#### Only fine-tune cross-attention layers

![](../assets/D2-46.png)     

> &#x2705; 由以上观察结果，finetune 时只更新 K 和 V 的参数。    

P47    
### How to prevent overfitting?    

> 用少量的数据finetune整个模型，容易造成过拟合。    
> &#x2705; 解决方法：通过在训练过程中引入一个正则化项来防止过拟合   

![](../assets/D2-47.png)   

> &#x2705; 从large scale image dataset中选择一些所标注文本与左图文本相似度比较高的图像。这些图像与文本的pair data用于计算正则化项。

P48   
### Personalized concepts   

#### 要解决的问题

> &#x2705; 目的：finetune SD 得到这只狗的文生图模型。  

How to describe personalized concepts?    

![](../assets/D2-48.png)   

> 但只有少量的关于这只狗的数据。

#### 解决方法

定义 V\\(^\ast \\) 为 modifier token in the text embedding space，例如：

> 解决方法：定义 \\(V^ \ast \\) 为 modifier token，并把它作为一个新的 token.  

V\\(^\ast \\) **dog**   

P49   

#### Pileline

Also fine-tune the modifier token V\\(^\ast \\) that describes the personalized concept   

![](../assets/D2-49.png)   

> &#x2705; 把 \\(V^ \ast \\) 代入 caption，并用这只狗的数据做 finetune。同样只更新 K 和 V.    


P50   
#### Single concept results

![](../assets/D2-50.png)   

P51   
### Multiple new concepts?    

#### 要解决的问题

![](../assets/D2-51.png)   

> 要生成同时包含moongate与这只狗的图像

P52   
#### Joint training

Combine the training dataset of multiple concepts    

![](../assets/D2-52.png)   

> &#x2705; 同时使用两个小样本数据 finetune，且使用 modifier token 和正则化图像，可以得到二者结合的效果。    

P53   
#### Two concept results

![](../assets/D2-53.png)   

> &#x2705; 也可以同时引入2个 modifier token．    

P54   
### Two concept results   

![](../assets/D2-54.png) 

Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    

P55   
## Key-Locked Rank One Editing for Text-to-Image Personalization   

![](../assets/D2-55.png)    

> Image Personalization任务

Tewel et al., <u>"Key-Locked Rank One Editing for Text-to-Image Personalization",</u> SIGGRAPH 2023    

#### 主要原理

> &#x2705; 方法：dynamic rank one update.     
> &#x2705; Perffusion 解决 Image Personalization 的 overfitting 问题的方法：  
> &#x2705; (1) 训练时，Introducing new *xxxx* that locks the new concepts cross-attention keys to their sub-ordinate category.    
> &#x2705; (2) 推断时，引入 a gate rank one approach 可用于控制 the learned concept的影响力。    
> &#x2705; (3) 允许 medel 把不同的 concept 结合到一起，并学到不同concept 之间的联系。    

#### Results

> 可以很好地model the interaction of the two conception。  

P56   
## T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models

![](../assets/D2-56-1.png)    

> 通过引入一个额外的apapter来增加对已有文生图模型的控制方法。  

Mou et al., <u>"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models",</u> arXiv 2023   

### Pipeline

![](../assets/D2-56-2.png)    

> &#x2705; Adapter 包含4个 feature extraction blocks 和3个 down sample blocks. 其中4个feature extraction block对应4个新增加的控制方法。3次降采样，对应于 UNET 的不同 Level.    

### 优点

> 这个方法具有以下优点：  

- Plug-and-play. Not affect original network topology and generation ability      

> 易使用

- Simple and small. ~77M parameters and ~300M storage    

> 简单、易训

- Flexible. Various adapters for different control conditions    
- Composable.  Several adapters to achieve multi-condition control    

> 不同的adaper可以combine，成为新的guidance。  

- Generalizable. Can be directly used on customed models    

P57   
### Result    

![](../assets/D2-57.png)    

> &#x2705; Adapter 可以使用于多种形式的 Control．     


P58   
## Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)

### ControlNet

> &#x2705; Control Net 是一种通过引入额外条件来控制 Diffusion Model 的网络架构。    

![](../assets/D2-58.png) 

> &#x2705; (a) 是预训练的 diffusion model. C 是要添加的新的condition.   
> &#x2705; 把 (a) 的网络复制一份，finetune copy 的网络，结果叠加。    
> &#x2705; Zero Convolution：1-1 卷积层，初始的 \\(w\\) 和 \\(b\\) 都为 0．   

Zhang and Agrawala, <u>"Adding Conditional Control to Text-to-Image Diffusion Models",</u> arXiv 2023    

P59   

### Control Net 应用到 Stable Diffusion 的例子

#### Pipeline

![](../assets/D2-59.png)   

P60   

#### Train objective   

$$
\mathcal{L} =\mathbb{E} _ {\mathbb{z}_0,t,\mathbf{c} _ t,\mathbf{c} _ f,\epsilon \sim \mathcal{N} (0,1)}[||\epsilon -\epsilon _\theta (\mathbf{z} _ t,t,\mathbf{c} _ t,\mathbf{c}_f)||^2_2] 
$$

where t is the time step, \\(\mathbf{c} _t\\) is the text prompts, \\(\mathbf{c} _ f\\) is the task-specific conditions    

> 需要（x, cf, ct）的pair data。  

P61    
### ControlNet Result

![](../assets/D2-61.png) 

P62   

![](../assets/D2-62.png) 

P63   

![](../assets/D2-63.png) 

P64   
## GLIGEN: Open-Set Grounded Text-to-Image Generation    

![](../assets/D2-64.png) 

Li et al., <u>"GLIGEN: Open-Set Grounded Text-to-Image Generation",</u> CVPR 2023        

> &#x2705; GLIGEN 是另一种网络架构。    

P65   

### Pipeline

GLIGEN对比ControlNet：

|GLIGEN|ControlNet|
|---|---|
|![](../assets/D2-65.png)|![](../assets/D2-65-2.png)|    
| &#x2705; 新增 Gated Self-Attention 层，加在 Attention 和 Cross Attention 之间。<br> &#x2705; GLIGEN 把 Condition 和 init feature 作 Concat  |&#x2705; Control Net 分别处理 feature 和 Control 并把结果叠加。    |

P66   
### GLIGEN Result

![](../assets/D2-66.png)  

P67   
# Other applications  

P68   
## Your Diffusion Model is Secretly a Zero-Shot Classifier

> &#x2705; 一个预训练好的 diffusion model （例如stable diffusion model），无须额外训练可以用作分类器，甚至能完成 Zero-shot 的分类任务。  

Li et al., <u>"Your Diffusion Model is Secretly a Zero-Shot Classifier",</u> arXiv 2023   

### Pipeline

![](../assets/D2-68.png)  

> &#x2705; 输入图像\\(x\\)，用随机噪声\\(\epsilon  \\)加噪；再用 condition c 预测噪声 \\(\epsilon  _\theta \\)。优化条件 C 使得 \\(\epsilon  _\theta \\) 最接近 \\(\epsilon \\). 得到的 C 就是分类。    


P69   
## Improving Robustness using Generated Data

> &#x2705; 使用 diffusion Model 做数据增强。    

![](../assets/D2-69.png)  

**Overview of the approach:**     
1. train a generative model and a non￾robust classifier, which are used to provide pseudo-labels to the generated data.    
2. The generated and original training data are combined to train a robust classifier.    

Gowal et al., <u>"Improving Robustness using Generated Data",</u> NeurIPS 2021    

P70  
## Better Diffusion Models Further Improve Adversarial Training   

![](../assets/D2-70.png)  

Wang et al., <u>"Better Diffusion Models Further Improve Adversarial Training",</u> ICML 2023    


P72   
# Reference   

 - Bao et al., <u>"All are Worth Words: a ViT Backbone for Score-based Diffusion Models",</u> arXiv 2022   
 - Peebles and Xie, <u>"Scalable Diffusion Models with Transformers",</u> arXiv 2022   
 - Bao et al., <u>"One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale",</u> arXiv 2023    
 - Jabri et al., <u>"Scalable Adaptive Computation for Iterative Generation",</u> arXiv 2022    
 - Hoogeboomet al., <u>"simple diffusion: End-to-end diffusion for high resolution images",</u> arXiv 2023   
 - Meng et al., <u>"SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations",</u> ICLR 2022    
 - Li et al., <u>"Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models",</u> NeurIPS 2022   
 - Avrahami et al., <u>"Blended Diffusion for Text-driven Editing of Natural Images",</u> CVPR 2022   
 - Hertz et al., <u>"Prompt-to-Prompt Image Editing with Cross-Attention Control",</u> ICLR 2023   
 - Kawar et al., <u>"Imagic: Text-Based Real Image Editing with Diffusion Models",</u> CVPR 2023   
 - Couairon et al., <u>"DiffEdit: Diffusion-based semantic image editing with mask guidance",</u> ICLR 2023   
 - Sarukkai et al., <u>"Collage Diffusion",</u>  arXiv 2023   
 - Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u>  ICML 2023   
 - Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion",</u> ICLR 2023     
 - Ruiz et al., <u>"DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation",</u> CVPR 2023    
 - Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u>  CVPR 2023   
 - Tewel et al., <u>"Key-Locked Rank One Editing for Text-to-Image Personalization",</u>  SIGGRAPH 2023   
 - Zhao et al., <u>"A Recipe for Watermarking Diffusion Models",</u>  arXiv 2023   
 - Hu et al., <u>"LoRA: Low-Rank Adaptation of Large Language Models",</u> ICLR 2022   
 - Li et al., <u>"GLIGEN: Open-Set Grounded Text-to-Image Generation",</u> CVPR 2023   
 - Avrahami et al., <u>"SpaText: Spatio-Textual Representation for Controllable Image Generation",</u> CVPR 2023   
 - Zhang and Agrawala, <u>"Adding Conditional Control to Text-to-Image Diffusion Models",</u> arXiv 2023    
 - Mou et al., <u>"T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models",</u> arXiv 2023   
 - Orgad et al., <u>"Editing Implicit Assumptions in Text-to-Image Diffusion Models",</u> arXiv 2023   
 - Han et al., <u>"SVDiff: Compact Parameter Space for Diffusion Fine-Tuning",</u> arXiv 2023   
 - Xie et al., <u>"DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter￾Efficient Fine-Tuning",</u> rXiv 2023   
 - Saharia et al., <u>"Palette: Image-to-Image Diffusion Models",</u> SIGGRAPH 2022   
 - Whang et al., <u>"Deblurring via Stochastic Refinement",</u> CVPR 2022   
 - Xu et al., <u>"Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models",</u> arXiv 2023   
 - Saxena et al., <u>"Monocular Depth Estimation using Diffusion Models",</u> arXiv 2023   
 - Li et al., <u>"Your Diffusion Model is Secretly a Zero-Shot Classifier",</u> arXiv 2023  
 - Gowal et al., <u>"Improving Robustness using Generated Data",</u> NeurIPS 2021   
 - Wang et al., <u>"Better Diffusion Models Further Improve Adversarial Training",</u> ICML 2023  

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/
