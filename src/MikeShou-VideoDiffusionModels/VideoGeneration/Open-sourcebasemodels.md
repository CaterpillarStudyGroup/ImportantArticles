P67   
![](../../assets/08-67.png)

P68   
# ModelScopeT2V

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|2023|Wang et al., “ModelScope Text-to-Video Technical Report,”|  

Leverage pretrained T2I models for video generation

 - Inflate Stable Diffusion to a 3D model, preserving pretrained weights   
 - Insert spatio-temporal blocks, **can handle varying number of frames**   

> &#x2705; 基本思路：(1) 以 Stable Diffusion 为基础，在 latent space 工作。 (2) 把 SD 中的 2D 操作扩展为 3D.   

|||
|--|--|
| ![](../../assets/08-68-1.png)  |  ![](../../assets/08-68-2.png) |

P69     
![](../../assets/08-69.png) 

> &#x2705; 扩展方法为 (2＋1)D，因此在 2D spatial 的卷积操作和 Attention 操作之后分别增加了 temporal 的卷积和 Attention.   

P70   

![](../../assets/08-70.png) 

P71   

Length = 1   
Model generate images   

> &#x2705; 时域卷积操作能指定 frame 数，因此可以“生成视频”与“生成图像”联合训练。   
> &#x2753; 时序卷积不能做流式，能不能用 transformer.   

P72   

![](../../assets/08-72.png) 


### ZeroScope

ZeroScope: finetunes ModelScope on a small set of high-quality videos, resulting into higher resolution at 1024 x 576, without the Shutterstock watermark    

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Wang et al., “ModelScope Text-to-Video Technical Report,”|  


> &#x2705; ZeroScope 在 ModelScope 上 finetune，使用了非常小但质量非常高的数据，得到了高分辨率的生成效果。   

P74  
### 其它相关工作

|||||
|--|--|--|--|
| " Robot dancing in times square,” arXiv 2023.  | " Clown fish swimming through the coral reef,” arXiv 2023.| " Melting ice cream dripping down the cone,” arXiv 2023.| " Hyper-realistic photo of an abandoned industrial site during a storm,” arXiv 2023.|
| ![](../../assets/08-74-1.png)  |  ![](../../assets/08-74-2.png) | ![](../../assets/08-74-3.png)  |  ![](../../assets/08-74-4.png) |


Wang et al., “ModelScope Text-to-Video Technical Report,” arXiv 2023.    



P80  
# VideoCrafter  

• Latent diffusion inserted with temporal layers

![](../../assets/08-80.png) 

Chen et al., “VideoCrafter1: Open Diffusion Models for High-Quality Video Generation,” arXiv 2023.    

P81  
# LaVie  

Joint image-video finetuning with curriculum learning

![](../../assets/08-81.png) 

Wang et al., “LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models,” arXiv 2023.   

> &#x2705; 提供了一套高质量数据集，生成的视频质量也更好（训练集很重要）。   
   

P84   
# Stable Video Diffusion   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|50|2023|Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets|Scaling latent video diffusion models to large datasets<br>**Data Processing and Annotation**||[link](https://caterpillarstudygroup.github.io/ReadPapers/50.html)|
||2024|CogVideo|


---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/