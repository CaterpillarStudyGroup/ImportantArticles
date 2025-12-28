输入：单/多视角视频  
输出：静态3DGS+GS的动态属性 或 动态3DGS  


|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
|176|2025.6.11|HAIF-GS: Hierarchical and Induced Flow-Guided Gaussian Splatting for Dynamic Scene|学习结构化且时间一致的运动表征|一个通过**稀疏锚点**驱动形变实现结构化一致动态建模的统一框架。<br>1. 通过锚点过滤器识别运动相关区域，抑制静态区域的冗余更新；2. 利用自监督诱导流引导变形模块，通过多帧特征聚合驱动锚点运动，无需显式光流标签；<br> 3. 为处理细粒度形变，分层锚点传播机制能依据运动复杂度提升锚点分辨率，并传播多级变换关系。    |运动信息来源：?<br>驱动方式：稀疏锚点驱动||
|173|2025.5.14|SplineGS: Learning Smooth Trajectories in Gaussian Splatting for Dynamic Scene Reconstruction|静态场景的高质量快速重建的基础上融入形变模块|用Spline来表征时间维度上的平滑形变|运动信息来源：单目视频<br>驱动方式：参数化线条驱动|
||2024.6.15|4d gaussian splatting for real-time dynamic scene rendering|[link](https://arxiv.org/pdf/2310.08528)|