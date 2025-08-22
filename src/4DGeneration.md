# 4D content generation

## 动态隐式表示 (基于 NeRF 的变体)

核心思想： 扩展静态 NeRF（学习从空间位置和视角到颜色/密度的映射），增加时间维度或变形场来建模动态。  
代表技术： 可变形 NeRF, 时变 NeRF。  
优点： 理论上能建模非常复杂、连续的动态效果（如流体、布料）。  
主要缺点：
- 优化时间长： 训练/优化过程非常耗时。
- 渲染效率低： 体渲染过程计算开销巨大。
- 重建质量受限： 由于优化和渲染的挑战，最终重建或生成的质量（清晰度、细节）可能不如人意。
- 与现代引擎兼容性差： 输出格式非标准网格/点云，难以集成到游戏/影视渲染管线。

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2024|Consistent4d: Consistent 360° dynamic object generation from monocular video|引入了一个视频到4D的框架，通过优化一个级联动态NeRF (Cascaded DyNeRF) 来从静态捕获的视频生成4D内容。|driving video|
|||Animate124 | 利用多种扩散先验，能够通过文本运动描述将单张野外图像动画化为3D视频。|SDS|
|||4D-fy| 使用混合分数蒸馏采样 (hybrid SDS)，基于多个预训练扩散模型实现了引人注目的文本到4D生成。|SDS|

## dynamic 3DGS

优点：
- 克服隐式方法（特别是动态 NeRF）的效率瓶颈和兼容性问题。  

主要缺点：
- 由于缺乏真实的4D标注数据，只能依赖多视角渲染进行监督学习，因此容易出现视角间的不一致性问题。

### 直接预测动态高斯属性

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|127|2025.7.31|Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis|直接对4D高斯进行diffusion生成数据量比较大，因此构建4D GS的VAE，并基于这个VAE进行隐空间的4G生成|[link](https://caterpillarstudygroup.github.io/ReadPapers/127.html)|

### 显式驱动静态高斯属性

核心思路： 利用控制点/蒙皮等显式或参数化结构来驱动显式图元（如高斯椭球）的变形，从而表示动态。这比纯隐式 NeRF 更高效且渲染质量更高。  
优点：
- 将静态几何与动态运动解耦。静态部分可以高效优化/表示，动态部分专注于运动。这通常比直接拟合整个时空函数更有效率。

主要缺点：
- 要解决如何有效控制显式图元随时间的变形以保持时空一致性和高质量。

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|129|2025.8.13|TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos|从视频中学习每个高斯点的动力学属性|开源|[link](https://caterpillarstudygroup.github.io/ReadPapers/129.html)|
||2025.6.11|HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene||    |[link](179.md)|
||2025.6.9|**PIG: Physically-based Multi-Material Interaction with 3D Gaussians**||    |[link](170.md)|
||2025.6.5|SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors||    |[link](159.md)|
||2025.6.4|**EnliveningGS: Active Locomotion of 3DGS**|| 3D 高斯溅射(3DGS)表示的 3D 模型能够实现主动运动   |[link](154.md)|
||2025.5.14|SplineGS: Learning Smooth Trajectories in Gaussian Splatting for Dynamic Scene Reconstruction|| 3DGS复杂场景重建 |[link](79.md)|
||2025|Stag4d: Spatial-temporal anchored generative 4d gaussians|
||2025|L4gm: Large 4d gaussian reconstruction model|
||2024|Dreammesh4d: Video-to-4d generation with sparse-controlled gaussian-mesh hybrid representation|
||2024|Animate3d: Animating any 3d model with multi-view video diffusion|
|111|2023.12|**Dreamgaussian4d: Generative 4d gaussian splatting**|1. 先使用DreamGaussianHD生成静态高斯溅射模型，然后通过基于六面体 (HexPlane) 的动态生成方法结合高斯变形技术生成动态内容 <br> 2. 运动信息来自driving video而不是video SDS|HexPlane显式驱动， 开源, driving video||[link](https://caterpillarstudygroup.github.io/ReadPapers/111.html)|

## Mesh Animation

### 先绑定再驱动

### 直接驱动

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|109|2025.6.11|**AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation**| 1. 将动态网格分解为初始状态与相对轨迹<br> 2. 融合网格拓扑信息 <br> 3. 基于注意力机制实现高效变长压缩与重建| 修正流，数据集   |[link](https://caterpillarstudygroup.github.io/ReadPapers/109.html)|
|110|2025.6.9|**Drive Any Mesh: 4D Latent Diffusion for Mesh Deformation from Video**|1. 以文本和目标视频为条件驱动Mesh<br> 2. 将动态网格分解为初始状态与相对轨迹 <br> 3. 使用latent set + Transformer VAE对动态Mesh进行编码<br> 4. 使用diffusion进行生成| Latent Sets，diffusion，数据集  |[link](https://caterpillarstudygroup.github.io/ReadPapers/110.html)|