![](./assets/d378e84bd11f484517ba2d687e8bb933_5_Table_1_-876463523.png)

# 无条件生成

### GAN

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2016| A deep learning framework for character motion synthesis and editing|深度学习框架	开创无条件运动生成的深度学习先河	生成多样性有限	奠定端到端生成基础|
||2023|Modi|Modi: Unconditional motion synthesis from diverse data.|StyleGAN风格迁移	将StyleGAN风格控制引入运动生成	模式崩溃/混合（生成动作重复或混乱）	风格化运动生成|

### VAE

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|14|2021|HuMoR: 3D Human Motion Model for Robust Pose Estimation|||[link](https://caterpillarstudygroup.github.io/ReadPapers/14.html)|
|19|2024|WANDR: Intention-guided Human Motion Generation|||[link](https://caterpillarstudygroup.github.io/ReadPapers/19.html)|

以下是整理后的表格，概述了各模型的架构、贡献、输入/输出及创新点：

| **模型名称**             | **基础架构**                | **主要贡献**                                                                 | **条件输入**          | **输出**                  | **训练目标**                                                                 | **关键创新点**                                                                 |
|--------------------------|-----------------------------|-----------------------------------------------------------------------------|-----------------------|---------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **ACTOR [116]** 2021         | Transformer + VAE           | 生成多样且真实的3D人体动作，作为后续研究的基线                              | 动作标签              | 多样化3D动作序列          | 潜在高斯分布对齐                                                            | 结合Transformer与VAE，支持从同一动作条件生成多动作变体                         |
| **TEMOS [117]**   2022       | 改进自ACTOR                 | 实现文本到SMPL动作的生成                                                    | 文本提示              | SMPL格式3D动作            | 共享潜在空间中文本与动作表征对齐（跨模态一致性）                            | 双对称编码器（动作序列+冻结DistilBERT文本编码器），共享潜在空间                 |
| **TEACH [118]**    2022      | 扩展自TEMOS                 | 处理连续文本指令生成连贯动作                                                | 文本序列              | 连贯3D动作序列            | 分层生成：非自回归（单个动作内） + 自回归（动作间时序组合）                 | 分层策略实现时序组合与平滑过渡                                                  |
| **T2M [119]**   2022         | 两阶段（卷积AE + 时序VAE）  | 分阶段生成文本对应动作                                                      | 文本特征              | 3D动作                    | 预训练运动编码器提取片段；时序VAE生成运动代码序列                           | 两阶段框架（先编码运动代码，再生成序列）                                        |
| **TMR [120]**   2023         | 改进自TEMOS                 | 提升文本-动作对齐，支持检索与生成                                           | 文本                  | 3D动作/跨模态检索结果     | 对比损失优化联合潜在空间，过滤误导性负样本（MPNet）                         | 引入CLIP式对比学习，优化负样本选择策略提升检索性能                              |

---

#### **关键说明**  
1. **条件输入与输出**：各模型均以文本或动作标签为条件生成动作，TEMOS与TMR兼容SMPL格式输出，T2M通过两阶段生成更灵活的运动片段组合。  
2. **架构演进**：  
   - ACTOR为基础模型，TEMOS引入文本编码，TEACH扩展至序列指令，TMR强化跨模态对齐。  
   - T2M采用独立的两阶段框架，与基于Transformer的模型形成对比。  
3. **训练策略**：  
   - TEMOS/TMR通过共享潜在空间实现跨模态对齐，TMR进一步引入对比损失提升检索能力。  
   - TEACH结合非自回归与自回归生成平衡质量与效率。  

此表格可快速对比各模型的核心特性及技术演进路径。

### VQ-VAE

以下是整理后的表格，概述了基于VQ-VAE的3D运动生成模型及其核心特性：

---

| **模型名称**         | **基础架构**                    | **主要贡献**                                                                 | **条件输入**      | **输出**              | **训练目标**                                                                 | **关键创新点**                                                                 |
|----------------------|---------------------------------|-----------------------------------------------------------------------------|-------------------|-----------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| **DiverseMotion [122]** 2023| VQ-VAE + 扩散模型              | 提升生成多样性与语义一致性                                                  | 文本提示（CLIP）  | 去噪后的运动令牌      | 扩散过程（前向破坏令牌，反向去噪）                                          | 用扩散过程替代自回归解码；引入分层语义聚合（HSA）增强文本语义理解               |
| **MoMask [123]**   2023  | VQ-VAE + 分层码本              | 分层生成粗糙到精细的运动细节                                                | 文本              | 分层运动令牌序列      | 掩码令牌建模（BERT风格） + 残差细化                                         | 分层码本结构；掩码预测生成粗糙运动，残差层逐步细化                              |
| **T2LM [124]**    2024   | 1D卷积VQ-VAE + Transformer     | 处理多句子文本生成长且复杂的动作序列                                        | 多句子文本        | 长动作序列            | 1D卷积压缩运动；Transformer编码文本时序关系                                  | 结合1D卷积与文本时序建模，实现跨动作平滑过渡                                    |

---

#### **关键说明**  
1. **技术路线**：  
   - **VQ-VAE核心思想**：将连续运动编码为离散令牌序列，类比语言建模，增强生成的结构性与可控性。  
   - **生成方式演进**：从自回归（T2M-GPT）→ 扩散（DiverseMotion）→ 分层残差（MoMask）→ LLM序列生成（MotionGPT）。  
2. **文本编码**：  
   - T2M-GPT与DiverseMotion依赖CLIP编码文本；后续模型（如MoMask、T2LM）直接学习端到端文本-运动映射。  
3. **效率与扩展性**：  
   - MotionGPT通过冻结VQ-VAE与LoRA技术，显著降低LLM微调成本，支持大规模文本-运动对齐。  
4. **应用方向**：  
   - 长序列生成（T2LM）、细节分层控制（MoMask）、多模态扩展（MotionGPT与LLM结合）是主要优化方向。  

此表格总结了基于VQ-VAE的运动生成模型的技术差异与演进趋势，突出离散化表示与语言建模结合的灵活性。

### Normalizing Flows.

### Diffusion Models



 Given the sequential nature of motion, transformers are often used within the denoising process to model temporal dependencies, though their integration varies significantly across models. Early adopters such as Flame [18], MotionDiffuse [ 22], and HMDM [126 ] all incorporate transformers for denoising, but differ in their conditioning strategies, temporal encoding, and loss functions. 

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2022|J. Kim, J. Kim, and S. Choi, "Flame: Free-form language-based motion synthesis & editing," arXiv preprint arXiv:2209.00349, 2022.|Flame [18] introduces a transformer-based motion decoder in place of the standard U-Net [231 ], using cross-attention to incorporate text features extracted with RoBERTa [322]. It introduces two special tokens for encoding motion length and diffusion timestep, both used during cross-attention to guide generation. |Flame|
||2022|MotionDiffuse [22 ]|MotionDiffuse [22 ], although similar in structure, handles the timestep differently by sampling it from a uniform distribution, and introduces variable-length motion generation by dividing sequences into sub-intervals. Each segment is paired with a corresponding text description, enabling part-wise and body-part-specific conditioning. It also incorporates Efficient Attention [323 ] to reduce computational cost and uses a classical transformer [ 5 ] for text encoding. While both Flame [18 ] and MotionDiffuse [22 ] rely on noise-based reconstruction, Flame adds a variational lower bound. In contrast, MotionDiffuse [ 22 ] optimizes only a mean squared error loss on the predicted noise. |
||2023|HMDM [126 ]|HMDM [126 ] takes a different approach by applying its primary reconstruction loss on the denoised signal rather than the noise. It encodes text using CLIP [7] and feeds the diffusion timestep into the transformer as a dedicated token, similar to Flame. However, HMDM [ 126 ] fixes the motion length and introduces a set of auxiliary loss functions designed to improve physical realism: a positional loss in joint 37 space, a velocity loss to enforce temporal consistency, and a foot contact loss defined using forward kinematics. These losses are combined with a standard reconstruction loss, forming a comprehensive objective that better preserves both spatial accuracy and motion smoothness. |
||2023|MakeAnAnimation [ 127 ]| In contrast, Departing from sequential generation, MakeAnAnimation [ 127 ] proposes a two-stage framework that first pre-trains on a large static 3D pose dataset, created from pose detection applied to image collections, to learn pose-text associations. Using a U-Net architecture [ 231 ] for the denoising network and a pre-trained T5 encoder [ 324 ] for text, the model generates full motion sequences concurrently. Unlike transformer-based models such as HMDM [126 ] and Flame [18 ], which enforce temporal consistency through specific loss functions, MakeAnAnimation [ 127 ] avoids such constraints and relies solely on standard diffusion loss. Despite this, it maintains motion continuity through its concurrent sampling and large-scale pre-training strategy. Recent works have also expanded the diffusion framework to support spatial and semantic constraints. |


### Motion Graph

### Regression

# TEXT-CONDITIONED MOTION GENERATION

## Action to Motion

### VQ-VAE

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|92|2025|Deterministic-to-Stochastic Diverse Latent Feature Mapping for Human Motion Synthesis|1. 第一阶段通，运动重建(VQVAE with different network)，学习运动潜在表征<br>2. 第二阶段，使用确定性特征映射过程(DerODE)构建高斯分布与运动潜在空间分布之间的映射关系<br>3. 生成时通过通过向确定性特征映射过程的梯度场中注入可控噪声(DivSDE)实现多样性。|VQVAE，新的生成模式|[link](https://caterpillarstudygroup.github.io/ReadPapers/92.html)|

### Diffusion

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|||MDM|	扩散模型（轨迹优化）	扩散模型首次应用于动作条件生成	多样性与保真度权衡（训练/采样轨迹曲线限制）	生成结果多样且逼真|
|||MLD|潜在空间DDPM	在潜在空间应用扩散模型，降低计算复杂度	与DDPM相同的采样效率问题	潜在空间压缩提升生成速度|

## Text to Motion

### VQ-VAE

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|88|2023|T2m-gpt: Generating human motion from textual descriptions with discrete representations|1. 首次将VQ-VAE引入运动生成，将运动建模为离散令牌序列<br> 2. 结合了**矢量量化变分自动编码器（VQ-VAE）**和**生成式预训练Transformer（GPT）**<br> 3. 生成质量(FID)有明显提升|VQ-VAE + Transformer, CLIP, 文本->Motion, 开源|[link](https://caterpillarstudygroup.github.io/ReadPapers/88.html)|

### Diffusion

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|

# 多模态动作生成

### VQ-VAE

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|87|2023.6.19|MotionGPT: Finetuned LLMs are General-Purpose Motion Generators|1. 利用VQ-VAE，将运动序列编码为一种特殊“语言”<br>2.  将运动生成视为序列到序列任务，结合LLM能力实现从文本到动作的端到端生成。<br>3. 首个多模态控制的动作生成方法|VQ-VAE + LLM + LoRA, 文本/key frame -> motion|[link](https://caterpillarstudygroup.github.io/ReadPapers/87.html)|

### Diffusion

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|85|2024|OmniControl: Control Any Joint at Any Time for Human Motion Generation|1. 使用ControlNet方式引入控制信号<br>2. 使用推断时损失注入方式进一步实现空间约束。|MDM，GMD，精确控制，ControlNet|[link](https://caterpillarstudygroup.github.io/ReadPapers/85.html)|
|86|2023|Guided Motion Diffusion for Controllable Human Motion Synthesis|将空间约束融入运动生成过程, 通过two-stage pipeline解决控制信号稀疏导致控制能力不足的问题。<br>第一阶段通过提升root投影轨迹loss强化轨迹控制，通过去噪函数实现稀疏轨迹->稠密轨迹的方法，从而生成稠密轨迹。<br>第二阶段使用稠密信号引导生成|GMD，轨迹控制|[link](https://caterpillarstudygroup.github.io/ReadPapers/86.html)|


# AUDIO-CONDITIONED MOTION GENERATION

## Music to Dance

## Speech to Gesture

# SCENE-CONDITIONED MOTION GENERATION

## Scene representation

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|29|2024|PACER+: On-Demand Pedestrian Animation Controller in Driving Scenarios|基于2D轨迹或视频的行人动作生成||[link](https://caterpillarstudygroup.github.io/ReadPapers/29.html)|

# Motion-Conditioned Motion Generation

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|27||Learning Human Motion from Monocular Videos via Cross-Modal Manifold Alignment|2D轨迹生成3D Motion||[link](https://caterpillarstudygroup.github.io/ReadPapers/27.html)

## Generation pipeline

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2024|WANDR: Intention-guided Human Motion Generation|基于初始与结束状态控制的动作生成。||[link](https://caterpillarstudygroup.github.io/ReadPapers/19.html)|

以下是整理后的表格，概述了3D人体运动生成与评估的数据集、关键指标及模型性能：

---

# **3D人体运动生成与合成数据集**

| **数据集名称**               | **关键统计**                                                                 | **模态**                          | **链接/备注**                     |
|------------------------------|-----------------------------------------------------------------------------|-----------------------------------|-----------------------------------|
| **Motion-X++ [301]**          | 1950万3D姿势，120,500序列，80,800视频，45,300音频，自由文本描述              | 3D/点云、文本、音频、视频         | [Motion-X++](https://mocap.cs.cmu.edu/) |
| **HumanMM (ms-Motion) [308]** | 120长序列（237分钟），600多视角视频重建，包含罕见交互动作                    | 3D/点云、视频                     | HumanMM                          |
| **Multimodal Anatomical [309]** | 51,051姿势（53解剖标记），48虚拟视角，2000+病理运动变体                      | 3D/点云、文本                     | Multimodal Anatomical Motion     |
| **AMASS [242]**               | 11,265动作片段（43小时），整合15个数据集（如CMU、KIT），SMPL格式，100+动作类别 | 3D/点云                           | [AMASS](https://amass.is.tue.mpg.de/) |
| **HumanML3D [119]**           | 14,616序列（28.6小时），44,970文本描述，200+动作类别                        | 3D/点云、文本                     | [HumanML3D](https://github.com/EricGuo5513/HumanML3D) |
| **BABEL [307]**               | 43小时动作（AMASS数据），250+动词中心动作类别，13,220序列，含时序动作边界    | 3D/点云、文本                     | [BABEL](https://babel.is.tue.mpg.de/) |
| **AIST++ [246]**              | 1,408舞蹈序列（1010万帧），9摄像机视角，15小时多视角视频                     | 3D/点云、视频                     | [AIST++](https://google.github.io/aichoreographer/) |
| **3DPW [245]**                | 60序列（51,000帧），多样化室内/室外场景，挑战性姿势与物体交互                | 3D/点云、视频                     | [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) |
| **PROX [310]**                | 20受试者，12交互场景，180标注RGB帧，场景感知运动分析                         | 3D/点云、图像                     | [PROX](https://prox.is.tue.mpg.de/) |
| **KIT-ML [304]**              | 3,911动作片段（11.23小时），6,278自然语言标注（52,903词），BVH/FBX格式       | 3D/点云、文本                     | [KIT-ML](https://motion-annotation.humanoids.kit.edu/) |
| **CMU MoCap**                 | 2605试验，6大类23子类，140+受试者                                           | 3D/点云、音频                     | [CMU MoCap](https://mocap.cs.cmu.edu/) |

---

# **文本到动作生成评估指标**

| **评估指标**           | **定义/计算方式**                                                                 | **用途**                          | **典型基准**                      |
|------------------------|---------------------------------------------------------------------------------|-----------------------------------|-----------------------------------|
| **FID (Fréchet Inception Distance)** | 比较生成与真实动作特征分布的Fréchet距离（低值表示更真实）                          | 真实性评估（如虚拟现实应用）       | HumanML3D, KIT Motion-Language    |
| **R-Precision [311]**  | 在共享嵌入空间中，正确文本在Top-k匹配中的比例（如Top-1/3）                       | 语义一致性（文本-动作对齐）        | HumanML3D, BABEL                   |
| **MultiModal Distance [312]** | 动作与文本嵌入的欧氏距离（低值表示强语义耦合）                                    | 跨模态语义对齐量化                 | ExpCLIP [41], TMR [120]            |
| **Diversity [313]**     | 随机采样动作对的平均距离（高值表示生成多样性）                                    | 动作空间覆盖广度                   | DiverseMotion [122], Motion Anything [125] |
| **Multimodality [313]** | 同一文本生成多动作的方差（高值表示单提示下的多样性）                              | 单提示多样性（避免重复）           | MoMask [123], TEACH [118]          |
| **用户研究 (User Studies)** | 人工评分自然度、情感表达、上下文相关性                                           | 主观质量评估（自动化指标补充）     | 研究论文中常用（如[314]）          |

---


# Reference

1. Generative AI for Character Animation: A Comprehensive Survey of Techniques, Applications, and Future Directions
2. Human Motion Generation Summary