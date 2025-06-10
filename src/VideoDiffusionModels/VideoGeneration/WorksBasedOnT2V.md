# Improved Text-2-Video

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|105|2025.5.27|Think Before You Diffuse: LLMs-Guided Physics-Aware Video Generation| 1. 使用LLM分析视频生成的预期效果，用于引导生成<br> 2. LLM对生成结果的评价也作为模型训练的Loss项<br> 3. 基于Wan大模型的LoRA微调| 数据集, LLM, LoRA, 数据集，物理  |[link](132.md)|

# Image-2-Video

## 强调符合物理规律

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2025.5.26|Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals|| 将物理力作为视频生成的控制信号的视频生成   |[link](125.md)|
|96|2025.3.26|**PhysAnimator: Physics-Guided Generative Cartoon Animation**|静态动漫插图生成动画<br>1. 分割出可形变部分<br>2. 转成2D Mesh<br>3. FEM驱动2D Mesh<br>4. 根据2D Mesh形变生成光流<br>5. 光流驱动Image草图<br>6. 草图作为控制信号，生成视频| 2D Mesh，FEM，ControlNet，光流，轨迹控制，SAM |[link](https://caterpillarstudygroup.github.io/ReadPapers/96.html)|

## 强调时序一致性

## 强调控制性

## 其它未归档

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2025.5.29|ATI: Any Trajectory Instruction for Controllable Video Generation||  视频生成中运动控制  |[link](138.md)|
||2025.5.26|MotionPro: A Precise Motion Controller for Image-to-Video Generation|| **通过交互式运动控制实现图像动画**   |[link](127.md)|

||2025.5.23|Temporal Differential Fields for 4D Motion Modeling via Image-to-Video Synthesis|| 通过图像到视频(I2V)合成框架来模拟规律的运动过程   |[link](117.md)|
||2025.5.20|LMP: Leveraging Motion Prior in Zero-Shot Video Generation with Diffusion Transformer|| 文+图像+运动视频->视频  |[link](101.md)|
||2025.5.14|CameraCtrl: Enabling Camera Control for Video Diffusion Models|| 相机位姿控制的视频生成 |[link](82.md)|
||2025.5.4|DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization|| 文生视频 |[link](46.md)|
||2025.5.1|T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation|| 文生视频，物理，评估 |[link](34.md)|
||2025.4.30|**Eye2Eye: A Simple Approach for Monocular-to-Stereo Video Synthesis**|| 文生3D视频 |[link](35.md)|
||2025.4.30|**ReVision: High-Quality, Low-Cost Video Generation with Explicit 3D Physics Modeling for Complex Motion and Interaction**||视频生成，物理  |[link](27.md)|
|97|2025|Draganything: Motion control for anything using entity representation|1. 分割可拖动对象<br> 2. 提取对象的latent diffusion feature<br> 3. 路径转为高斯热图<br> 4. feature和heatmap作为控制信号进行生成|轨迹控制，ControlNet，高斯热图，SAM，潜在扩散特征|[link](https://caterpillarstudygroup.github.io/ReadPapers/97.html)|
||2025|Sparsectrl: Adding sparse controls to text-to-video diffusion models|深度控制|
||2024|Cinemo: Consistent and controllable image animation with motion diffusion models|
||2024.06|Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance|pose控制|
|47|2024|Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics|拖拽控制的对象零件级运动的视频生成|零件级运动数据集|[link](https://caterpillarstudygroup.github.io/ReadPapers/47.html)|
||2024|Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality|
||2024| Physgaussian: Physicsintegrated 3d gaussians for generative dynamics|
||2025|Physdreamer: Physics-based interaction with 3d objects via video generation|