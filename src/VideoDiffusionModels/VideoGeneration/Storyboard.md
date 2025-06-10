P108 
# 2.5 Storyboard
 
P113   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|84|2024|Learning Long-form Video Prior via Generative Pre-Training|利用GPT生成长视频内容的结构化信息，用于帮助下游的视频生成/理解任务。|结构化信息，数据集|[dataset](<https://github.com/showlab/Long-form-Video-Prior>) <br>  [link](https://caterpillarstudygroup.github.io/ReadPapers/84.html) |
|61|2023|Xie et al., “VisorGPT: Learning Visual Prior via Generative Pre-Training,”|A “diffusion over diffusion” architecture for very long video generation ||[link](https://caterpillarstudygroup.github.io/ReadPapers/61.html)|
||2023|Lin et al., “VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning,”|Use storyboard as condition to generate video<br> &#x2705; Control Net，把文本转为 Pixel 图片。|![](../../assets/08-121.png) ![](../../assets/08-122.png) |


|||
|--|--|
|  ![](../../assets/08-125-1.png) | **Dysen-VDM** (Fei et al.)<br>Storyboard through scene graphs<br>“Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models,” arXiv 2023. |
| ![](../../assets/08-125-2.png)  | **DirectT2V** (Hong et al.) <br> Storyboard through bounding boxes <br> “Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation,” arXiv 2023. |
|  ![](../../assets/08-125-3.png)  | **Free-Bloom** (Huang et al.)<br>Storyboard through detailed text prompts<br> “Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator,” NeurIPS 2023. |
|  ![](../../assets/08-125-4.png) | **LLM-Grounded Video Diffusion Models** (Lian et al.) <br> Storyboard through foreground bounding boxes <br> “LLM-grounded Video Diffusion Models,” arXiv 2023. |

P104

> &#x2705; 生成电影级别的视频，而不是几秒钟的视频。   

P106   
> &#x2705; 文本 → 结构化的中间脚本 → 视频  

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/