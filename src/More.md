# 未归档论文

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|39|2024|Motion Avatar: Generate Human and Animal Avatars with Arbitrary Motion|文生3D Mesh + 文生成3D动作 + 重定向 = 3D动物运动序列||[link](https://caterpillarstudygroup.github.io/ReadPapers/39.html)|
|35||MagicPony: Learning Articulated 3D Animals in the Wild|图像生成3D动物Mesh并绑定，图像生成3D动作||[link](https://caterpillarstudygroup.github.io/ReadPapers/35.html)|
|32||Artemis: Articulated Neural Pets with Appearance and Motion Synthesis|NGI 动物的高度逼真渲染||[link](https://caterpillarstudygroup.github.io/ReadPapers/32.html)|
|30|2024|CAT3D: Create Anything in 3D with Multi-View Diffusion Models|基于Diffusion的3D重建||[link](https://caterpillarstudygroup.github.io/ReadPapers/30.html)|

# [2025] Ouroboros-Diffusion: Exploring Consistent Content Generation in Tuning-free Long Video Diffusion

### 翻译：
**基于预训练文生视频模型的先进先出（FIFO）视频扩散方法**，近期已成为无需微调的长视频生成有效方案。该技术通过维护一个噪声逐步递增的视频帧队列，在队列头部持续输出干净帧，同时在尾部入队高斯噪声。然而，由于缺乏跨帧的对应关系建模，**FIFO-Diffusion往往难以维持生成视频的长程时间一致性**。本文提出**衔尾蛇扩散（Ouroboros-Diffusion）**——一种新型视频去噪框架，旨在增强结构与内容（主体）一致性，实现任意长度视频的连贯生成。具体而言：
1. **队列尾部潜在空间采样技术**：通过改进队列尾部的潜在空间采样策略，增强结构一致性，确保帧间感知平滑过渡；
2. **主体感知跨帧注意力机制（SACFA）**：在短片段内对齐跨帧主体，提升视觉连贯性；
3. **自循环引导技术**：利用队列前端所有历史干净帧的信息，指导尾部含噪帧的去噪过程，促进丰富且有上下文关联的全局信息交互。  
在VBench基准测试上的长视频生成实验表明，Ouroboros-Diffusion在**主体一致性、运动平滑性、时间一致性**等关键指标上显著优于现有方法，展现出全面优越性。

---

### 关键术语对照：
- **FIFO (First-In-First-Out)** → 先进先出  
- **Tuning-free long video generation** → 无需微调的长视频生成  
- **Long-range temporal consistency** → 长程时间一致性  
- **Ouroboros-Diffusion** → 衔尾蛇扩散（保留英文术语，体现自循环特性）  
- **Subject-Aware Cross-Frame Attention (SACFA)** → 主体感知跨帧注意力机制（SACFA）  
- **Self-recurrent guidance** → 自循环引导  
- **VBench benchmark** → VBench基准测试

# [2025] RealVVT: Towards Photorealistic Video Virtual Try-on via Spatio-Temporal Consistency

### 翻译：
**虚拟试穿技术作为计算机视觉与时尚领域的交叉核心任务，旨在通过数字手段模拟服饰在人体上的穿着效果**。尽管单图像虚拟试穿（VTO）已取得显著进展，但现有方法往往难以在长视频序列中保持服饰外观的**一致性与真实性**，其根源在于动态人体姿态捕捉与目标服饰特征维持的复杂性。我们基于现有视频基础模型提出**RealVVT**——一种**逼真视频虚拟试穿框架**，专门针对动态视频场景下的稳定性与真实感进行强化。该方法包含三部分核心技术：
1. **服饰与时序一致性策略**：确保跨帧服饰纹理、褶皱等细节的连续性；
2. **无关性引导的注意力聚焦损失机制**：通过约束无关区域特征，强化空间一致性；
3. **姿态引导的长视频VTO技术**：适配长视频序列处理，优化动态试穿的流畅性。  
通过在多数据集上的广泛实验验证，RealVVT在单图像与视频VTO任务中均超越现有最优模型，为时尚电商与虚拟试衣场景提供了实用化解决方案。

---

### 关键术语对照：
- **Virtual try-on (VTO)** → 虚拟试穿（VTO）  
- **PhotoRealistic Video Virtual Try-on** → 逼真视频虚拟试穿  
- **Clothing & Temporal Consistency** → 服饰与时序一致性  
- **Agnostic-guided Attention Focus Loss** → 无关性引导的注意力聚焦损失  
- **Pose-guided Long Video VTO** → 姿态引导的长视频虚拟试穿  
- **Fashion e-commerce** → 时尚电商

# [2025] FlexiClip: Locality-Preserving Free-Form Character Animation

**为剪贴画图像赋予流畅运动的同时保持视觉保真度与时间连贯性是一项重大挑战**。现有方法（如AniClipart）虽能有效建模空间形变，却常难以确保平滑的时序过渡，导致动作突变和几何失真等伪影。类似地，文本到视频（T2V）和图像到视频（I2V）模型因自然视频与剪贴画风格的统计特性差异而难以处理此类内容。本文提出**FlexiClip**——一种新方法，通过协同解决时序一致性与几何完整性的交织难题来突破这些限制。FlexiClip在传统**贝塞尔曲线轨迹建模**的基础上引入三项关键创新：
1. **时序雅可比矩阵**：通过增量式修正运动动力学，确保动作连贯性；
2. **基于概率流常微分方程（pfODEs）的连续时间建模**：降低时序噪声对生成质量的影响；
3. **受GFlowNet启发的流匹配损失**：优化运动过渡的平滑性。  
这些改进使得FlexiClip能在快速运动和非刚性形变等复杂场景下生成连贯动画。大量实验验证了FlexiClip在生成**流畅自然且结构一致**的动画效果上的有效性（涵盖人类、动物等多样剪贴画类型）。通过将时空建模与预训练视频扩散模型结合，FlexiClip为高质量剪贴画动画树立了新标杆，并在广泛视觉内容上展现出鲁棒性能。  
**项目主页**：https://creative-gen.github.io/flexiclip.github.io/

---

### 关键术语对照：
- **Temporal coherence** → 时间连贯性  
- **Bézier curve-based trajectory modeling** → 基于贝塞尔曲线的轨迹建模  
- **Temporal Jacobians** → 时序雅可比矩阵  
- **Probability flow ODEs (pfODEs)** → 概率流常微分方程（pfODEs）  
- **Flow matching loss** → 流匹配损失  
- **Non-rigid deformations** → 非刚性形变