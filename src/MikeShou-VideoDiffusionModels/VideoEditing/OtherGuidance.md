
P263   
# 3 Video Editing  

## 3.5 Other Guidance


P264  

![](../../assets/08-264.png)   

> &#x2705; 在已有图片的情况，直接输入 Prompt 不符合用户习惯，用户只需描述要修改的点，通过 Prompt 2 Prompr 转化为完整 prompt.    

P265   
## InstructPix2Pix

Instruction-guided image editing

![](../../assets/08-265.png) 

Brooks et al., “InstructPix2Pix: Learning to Follow Image diting Instructions,” CVPR 2023.  


P266   
## InstructVid2Vid

Instruction-guided Video Editing

 - Generate ⟨instruction, video⟩ dataset using ChatGPT, BLIP and Tune-A-Video   
 - Train inflated Stable Diffusion for instruction-guided video editing   

![](../../assets/08-266.png) 

Qin et al., “InstructVid2Vid: Controllable Video Editing with Natural Language Instructions,” arXiv 2023.    


> &#x2705;（1）把说话的部分 mask 掉 （2）用 diffusion 根据 Audio Feature 生成说话的部分。   
> &#x2705; 额外约束：（1）reference 状态 （2）前后帧 smooth     
> &#x2705; 语音驱动嘴形。   


P267   
## Speech Driven Video Editing via an Audio-Conditioned Diffusion Model

Speech-driven video editing

![](../../assets/08-267.png) 

Bigioi et al., “Speech Driven Video Editing via an Audio-Conditioned Diffusion Model,” arXiv 2023.   

P268   
## Soundini

Sound-guided video editing

![](../../assets/08-268.png) 

Lee et al., “Soundini: Sound-Guided Diffusion for Natural Video Editing,” arXiv 2023.    

P269   
## Video Editing Under Various Guidance: More Works

|||
|--|--|
| ![](../../assets/08-269-1.png)  | **Collaborative Score Distillation** (Kim et al.) <br> Instruction-guide video editing <br> “Collaborative Score Distillation for Consistent Visual Synthesis,” NeurIPS 2023. |
| ![](../../assets/08-269-2.png)  | **Make-A-Protagonist** (Zhao et al.) <br> Video ediSng with an ensemble of experts <br> “Make-A-Protagonist: Generic Video Edigng with An Ensemble of Experts,” arXiv 2023. |
| ![](../../assets/08-269-3.png)  | **DragNUWA** (Yin et al.) <br> Multimodal-guided video editing <br> “DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory,” arXiv 2023. |

P272

> &#x2705; showlab/Awesome-Video-Diffusion    

