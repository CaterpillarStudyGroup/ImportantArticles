P9    
# 图像编辑

P10    
## Gaussian Noise方法

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2022|SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations||[link](https://caterpillarstudygroup.github.io/ReadPapers/23.html)|

## DDIM Inversion方法

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|23|2023|Dual diffusion implicit bridges for image-to-image translation||[link](https://caterpillarstudygroup.github.io/ReadPapers/23.html)|
|24|2023|DiffEdit: Diffusion-based semantic image editing with mask guidance||[link](https://caterpillarstudygroup.github.io/ReadPapers/24.html)|
|25|2023|Imagic: Text-Based Real Image Editing with Diffusion Models||[link](https://caterpillarstudygroup.github.io/ReadPapers/25.html)|

## Attention based 方法

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Prompt-to-Prompt Image Editing with Cross-Attention Control|通过控制生成过程中的 attention maps进行图像编辑|attention控制|[link](https://caterpillarstudygroup.github.io/ReadPapers/20.html)|
||2023|NULL-text Inversion for Editing Real Images Using Guided Diffusion Models|针对真实图像（非生成图像）的编辑，以[CFG](https://caterpillarstudygroup.github.io/ReadPapers/6.html)为基础，fix condition分支，优化无condition分支，使其embedding向condition分支的embedding靠近|attention控制|
|||Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation|在上一篇的基础上，通过attention注入的方式加速上述流程|attention控制|
||2023|InstructPix2Pix: Learning to Follow Image Editing Instructions|在上一篇的基础上，通过attention注入的方式加速上述流程|attention控制|


P32   
# 特定对象定制化的图像生成   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|52|2024|Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models|多个特定对象的图像生成，让多个特定的对象生成到一张图像中，并用2D pose控制对象的动作|TI, LoRA|[link](https://caterpillarstudygroup.github.io/ReadPapers/51.html)|
|62|2023|Ruiz et al., “DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation,” ||| [link](https://caterpillarstudygroup.github.io/ReadPapers/62.html)|
|63|2023|Gal et al., <u>"An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion"|||   [link](https://caterpillarstudygroup.github.io/ReadPapers/63.html)|
|**38**|2021|Lora: Low-rank adaptation of large language models|对已训好的大模型进行微调，生成想要的风格。学习其中的残差。残差通常可以用low rank Matrix来拟合，因此称为low-rank adaptation。low rank的好处是要训练或调整的参数非常少。||[link](https://caterpillarstudygroup.github.io/ReadPapers/38.html)|
|||Lora + Dreambooth (by Simo Ryu)||| <https://github.com/cloneofsimo/lora> |
||2023|Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models|将多个LoRA融合到一个模型时，解决LoRA之间的冲突问题。|

P43
# 多个特定对象定制化的图像生成   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|52|2024|Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models|多个特定对象的图像生成，让多个特定的对象生成到一张图像中，并用2D pose控制对象的动作|TI, LoRA|[link](https://caterpillarstudygroup.github.io/ReadPapers/51.html)|



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

## ControlNet    

> &#x2705; Control Net 是一种通过引入额外条件来控制 Diffusion Model 的网络架构。 

![](../assets/08-28.png)

> &#x2705; 方法：(1) 预训练好 Diffusion Model (2) 参数复制一份，原始网络 fix (3) 用各种 condition finetune 复制出来的网络参数。 (4) 两个网络结合到一起。   

![](../assets/D2-58.png) 

> &#x2705; (a) 是预训练的 diffusion model. C 是要添加的新的condition.   
> &#x2705; 把 (a) 的网络复制一份，finetune copy 的网络，结果叠加。    
> &#x2705; Zero Convolution：1-1 卷积层，初始的 \\(w\\) 和 \\(b\\) 都为 0．   

以Stable Diffusion为例来说明ControlNet的用法。

![](../assets/08-27.png)



$$
\mathcal{L} =\mathbb{E} _ {\mathbb{z}_0,t,\mathbf{c} _ t,\mathbf{c} _ f,\epsilon \sim \mathcal{N} (0,1)}[||\epsilon -\epsilon _\theta (\mathbf{z} _ t,t,\mathbf{c} _ t,\mathbf{c}_f)||^2_2] 
$$

where t is the time step, \\(\mathbf{c} _t\\) is the text prompts, \\(\mathbf{c} _ f\\) is the task-specific conditions    

> 需要（x, cf, ct）的pair data。  

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2013|Adding Conditional Control to Text-to-Image Diffusion Models|

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

## Imagen
Imagen: Saharia et al., “Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding,” arXiv 2022.    
  

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
