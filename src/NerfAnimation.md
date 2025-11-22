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
||2025.6.17|GAF: Gaussian Action Field as a Dvnamic World Model for Robotic Mlanipulation||    |[link](198.md)|
||2024|Consistent4d: Consistent 360° dynamic object generation from monocular video|引入了一个视频到4D的框架，通过优化一个级联动态NeRF (Cascaded DyNeRF) 来从静态捕获的视频生成4D内容。|driving video|
|||Animate124 | 利用多种扩散先验，能够通过文本运动描述将单张野外图像动画化为3D视频。|SDS|
|||4D-fy| 使用混合分数蒸馏采样 (hybrid SDS)，基于多个预训练扩散模型实现了引人注目的文本到4D生成。|SDS|
