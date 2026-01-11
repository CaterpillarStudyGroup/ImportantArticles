|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
||2026.1.8|Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video|一种用于单目4D网格重建的前馈模型。给定动态对象的单目视频，我们的模型能够重建对象的完整3D形状与运动，并表示为变形场。|Feed-forward|[link](https://arxiv.org/pdf/2601.05251)|
||2025.8.27|ScanMove: Motion Prediction and Transfer for Unregistered Body Meshes|未注册未绑定的人体Mesh难以直接驱动|运动嵌入网络+逐顶点特征场，生成驱动网格变形的时空变形场。|
||2025.6.18|GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects||    |[link](201.md)|
|109|2025.6.11|**AnimateAnyMesh: A Feed-Forward 4D Foundation Model for Text-Driven Universal Mesh Animation**| 1. 将动态网格分解为初始状态与相对轨迹<br> 2. 融合网格拓扑信息 <br> 3. 基于注意力机制实现高效变长压缩与重建| 修正流，数据集   |[link](https://caterpillarstudygroup.github.io/ReadPapers/109.html)|
|110|2025.6.9|**Drive Any Mesh: 4D Latent Diffusion for Mesh Deformation from Video**|1. 以文本和目标视频为条件驱动Mesh<br> 2. 将动态网格分解为初始状态与相对轨迹 <br> 3. 使用latent set + Transformer VAE对动态Mesh进行编码<br> 4. 使用diffusion进行生成| Latent Sets，diffusion，数据集  |[link](https://caterpillarstudygroup.github.io/ReadPapers/110.html)|