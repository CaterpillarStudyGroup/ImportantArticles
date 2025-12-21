# Image（提供动作信息）Text(提供外观信息)-2-Video

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|126|2025.7.22|MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation|1. 参考对象（**动作信息来自图像**）与目标对象（**外观信息来自文本**）外观或结构差异显著<br> 2. 显示提取源和目标在外观上的语义匹配以及对应部分的形变关系，通过对源做warp得到目标的大致轮廓，以引作为condition引入视频生成|training-free，开源|

# Image（提供外观信息）-2-Video

## 强调符合物理规律

1. 如何描述物理规律：LLM对物理的理解、特定的数据集、已有的物理模型  
2. 如何使用物理规律：数据集、损失
3. 是否显示提取物理规律

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|106|2025.5.26|Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals|1. 将物理力(全局力和点力)编码后作为生成条件引导生成<br> 2. 构造少量数据集 <br> 3. 证明大TI2V模型 + 少量样本能得到比较好的泛化性 |开源， CogVideoX + ControlNet，物理 |[link](https://caterpillarstudygroup.github.io/ReadPapers/106.html)|
||2025.5.1|T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation|| 文生视频，物理，评估 |[link](34.md)|
|96|2025.3.26|**PhysAnimator: Physics-Guided Generative Cartoon Animation**|静态动漫插图生成动画<br>1. 分割出可形变部分<br>2. 转成2D Mesh<br>3. FEM驱动2D Mesh<br>4. 根据2D Mesh形变生成光流<br>5. 光流驱动Image草图<br>6. 草图作为控制信号，生成视频| 2D Mesh，FEM，ControlNet，光流，轨迹控制，SAM |[link](https://caterpillarstudygroup.github.io/ReadPapers/96.html)|
||2025|Physdreamer: Physics-based interaction with 3d objects via video generation|
||2024.9.27|PhysGen| 通过刚体物理仿真将单张图像与输入力转换为真实视频，证明从视觉数据推理物理参数的可能性；

## 强调时序一致性

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|130|2025.8.25|Multi-Object Sketch Animation with Grouping and Motion Trajectory Priors|

## 强调控制性

1. 如何对控制信号进行表示
2. 如何注入控制信号

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|97|2025|Draganything: Motion control for anything using entity representation|1. 分割可拖动对象<br> 2. 提取对象的latent diffusion feature<br> 3. 路径转为高斯热图<br> 4. feature和heatmap作为控制信号进行生成|轨迹控制，ControlNet，高斯热图，SAM，潜在扩散特征|[link](https://caterpillarstudygroup.github.io/ReadPapers/97.html)|
|47|2024|Puppet-Master: Scaling Interactive Video Generation as a Motion Prior for Part-Level Dynamics|拖拽控制的对象零件级运动的视频生成|零件级运动数据集|[link](https://caterpillarstudygroup.github.io/ReadPapers/47.html)|

## 其它未归档

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2025.6.17|VideoMAR: Autoregressive Video Generatio with Continuous Tokens||    |[link](200.md)|
||2025.5.29|ATI: Any Trajectory Instruction for Controllable Video Generation||  视频生成中运动控制  |[link](138.md)|
||2025.5.26|MotionPro: A Precise Motion Controller for Image-to-Video Generation|| **通过交互式运动控制实现图像动画**   |[link](127.md)|
||2025.5.23|Temporal Differential Fields for 4D Motion Modeling via Image-to-Video Synthesis|| 通过图像到视频(I2V)合成框架来模拟规律的运动过程   |[link](117.md)|
||2025.5.20|LMP: Leveraging Motion Prior in Zero-Shot Video Generation with Diffusion Transformer|| 文+图像+运动视频->视频  |[link](101.md)|
||2025.5.14|CameraCtrl: Enabling Camera Control for Video Diffusion Models|| 相机位姿控制的视频生成 |[link](82.md)|
||2025.5.4|DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization|| 文生视频 |[link](46.md)|
||2025.4.30|**Eye2Eye: A Simple Approach for Monocular-to-Stereo Video Synthesis**|| 文生3D视频 |[link](35.md)|
||2025|Sparsectrl: Adding sparse controls to text-to-video diffusion models|深度控制|
||2024|Cinemo: Consistent and controllable image animation with motion diffusion models|
||2024.06|Mimicmotion: High-quality human motion video generation with confidence-aware pose guidance|pose控制|
||2024|Vr-gs: A physical dynamics-aware interactive gaussian splatting system in virtual reality|

