### 直接预测动态高斯属性

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|127|2025.7.31|Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis|直接对4D高斯进行diffusion生成数据量比较大，因此构建4D GS的VAE，并基于这个VAE进行隐空间的4G生成|[link](https://caterpillarstudygroup.github.io/ReadPapers/127.html)|
||2025.6.5|SinGS: Animatable Single-Image Human Gaussian Splats with Kinematic Priors||    |[link](159.md)|
||2024.6.14|L4gm: Large 4d gaussian reconstruction model|单视角视频输入生成动态物体的4D大重建模型|1. 多视角视频数据集<br> 2.基于预训练的3D大重建模型LGM, 通过低帧率采样的视频帧生成逐帧的3D高斯泼溅表征|[link](https://arxiv.org/pdf/2406.10324)|
||2023.22|Stag4d: Spatial-temporal anchored generative 4d gaussians|实现具有时空一致性的高保真4D生成|单目视频->多目视频，SDS优化出GS属性|[link](https://arxiv.org/pdf/2403.14939)|
|36|2023.4|GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians|1. 引入可动画化的 3D GS 来明确代表各种姿势和服装风格的人类。<br> 2. 设计一个动态外观网络以及一个可优化的特征张量，用于实现运动到外观的映射。通过动态属性进一步增强3D GS表示。<br> 3. 对运动和外观进行联合优化，缓解『单目视频中运动估计不准确』的问题。 |开源、SMPLX、动态高斯|[link](https://caterpillarstudygroup.github.io/ReadPapers/36.html)|