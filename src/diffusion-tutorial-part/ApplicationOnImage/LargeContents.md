P68   
# Diffusion Models for Large Contents   

同样的方法也可用于Applications such as long images, looped motion, 360 images…   

 - Suppose model is trained on small, squared images, how to extend it to larger images?   
 - Outpainting is always a solution, but not a very efficient one!   

Let us generate this image with a diffusion model only trained on squared regions:    

![](../../assets/D3-68-1.png)    

1. Generate the center region \\(q(\mathbf{x} _ 1,\mathbf{x} _ 2)\\)    
2. Generate the **surrounding region conditioned on parts of the center image** \\(q(\mathbf{x} _ 3|\mathbf{x} _ 2)\\)    

![](../../assets/D3-68-2.png)    

Latency scales linearly with the content size!     

> &#x2705; 根据左边的图生成右边的图，存在的问题：慢     
> &#x2705; 直接生成大图没有这样的数据。   
> &#x2705; 并行化的生成。    
   
P69   
## DiffCollage

 - Unlike autoregressive models, diffusion models can generate large contents **in parallel**!    

![](../../assets/D3-69-1.png)    

P70     

 - A “large” diffusion model from “small” diffusion models!   

![](../../assets/D3-70-1.png)    

![](../../assets/D3-70-2.png)    

P71   

## More Works

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models"||
||2023|Jiménez, <u>"Mixture of Diffusers for scene composition and high resolution image generation",</u> arXiv 2023| - Based on similar ideas but differ in how overlapping regions are mixed.<br> &#x2705; 这种并行化方法可以用于各种 overlapping 的场景。 |
||2023|Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> ICML 2023 |

   



---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/