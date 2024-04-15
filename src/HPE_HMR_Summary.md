# Human Pose Estimation

> 输出关节位置、旋转、连接关系

## 单人HPE
### 图像单人HPE
#### Solving Depth Ambiguity
#### Solving Body Structure Understanding
#### Solving Occlusion Problems
#### Solving Data Lacking
### 视频单人HPE
#### Solving Single-frame Limitation
#### Solving Real-time Problems
#### Solving Body Structure Understanding
#### Solving Occlusion Problems
#### Solving Data Lacking
## 多人HPE
# Human Mesh Recovery
## Template-based human mesh recovery
### Naked human body recovery
#### **Multimodal Methods**

|ID|Year|Name|link|
|---|---|---|---|
|[123]|2019|
|[124]|2022|
|[125]|2022|
|[126]|2022|

#### Utilizing Attention Mechanism
#### **Exploiting Temporal Information**

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|[134]|2019|
|[135]|2021|
|[136]|2021|
|[137]|2021|
|[138]|2022|
|[139]|2023|Global-to-local modeling for video-based 3d human pose and shape estimation|To effec- tively balance the learning of short-term and long-term temporal correlations, Global-to-Local Transformer (GLoT) [139] structurally decouples the modeling of long-term and short-term correlations.|视频，单人，SMPL，非流式，transformer|https://caterpillarstudygroup.github.io/ReadPapers/12.html|

#### Multi-view Methods.
#### Boosting Efficiency
#### Developing Various Representations
#### Utilizing Structural Information
#### Choosing Appropriate Learning Strategies

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|161|2019|
|44|2020|
|163|2020|Coherent reconstruction of multiple humans from a single image||图像，多人|
|164|2021|
|46|2021|
|214|2021|
|165|2022|
|166|2022|
|167|2023|Jotr: 3d joint con- trastive learning with transformers for occluded human mesh recovery|fuses 2D and 3D features and incorporates supervision for the 3D feature through a Transformer-based contrastive learning framework||
|162|2023|Refit: Recurrent fitting network for 3d human recovery|reprojects keypoints and refines the human model via a feedback-update loop mechanism||
|4|2023|Co-evolution of pose and mesh for 3d human body estimation from video|introduced a co-evolution method for human mesh recovery that utilizes 3D pose as an intermediary. This method divides the process into two distinct stages: initially, it estimates the 3D human pose from video, and subsequently, it regresses mesh vertices based on the estimated 3D pose, combined with temporal image features|开源|
|168|2023|Cyclic test-time adaptation on monocular video for 3d human mesh reconstruction|To bridge the gap between training and test data, CycleAdapt [168] proposed a domain adaptation method including a mesh reconstruction network and a motion denoising network en- abling more effective adaptation.||

### Detailed human body recovery
## Template-free human body recovery
