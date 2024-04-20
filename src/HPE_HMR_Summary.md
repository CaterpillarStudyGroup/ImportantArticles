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
|[139]|2023|Global-to-local modeling for video-based 3d human pose and shape estimation|To effec-tively balance the learning of short-term and long-term temporal correlations, Global-to-Local Transformer (GLoT) [139] structurally decouples the modeling of long-term and short-term correlations.|视频，单人，SMPL，非流式，transformer|[link](https://caterpillarstudygroup.github.io/ReadPapers/12.html)|


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
|167|2023|Jotr: 3d joint con-trastive learning with transformers for occluded human mesh recovery|fuses 2D and 3D features and incorporates supervision for the 3D feature through a Transformer-based contrastive learning framework||
|162|2023|Refit: Recurrent fitting network for 3d human recovery|reprojects keypoints and refines the human model via a feedback-update loop mechanism||
|4|2023|Co-evolution of pose and mesh for 3d human body estimation from video|introduced a co-evolution method for human mesh recovery that utilizes 3D pose as an intermediary. This method divides the process into two distinct stages: initially, it estimates the 3D human pose from video, and subsequently, it regresses mesh vertices based on the estimated 3D pose, combined with temporal image features|开源、单人、视频|[link](https://caterpillarstudygroup.github.io/ReadPapers/13.html)
|168|2023|Cyclic test-time adaptation on monocular video for 3d human mesh reconstruction|To bridge the gap between training and test data, CycleAdapt [168] proposed a domain adaptation method including a mesh reconstruction network and a motion denoising network en-abling more effective adaptation.||

### Detailed human body recovery

#### With Clothes

#### With Hands

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|173|2023|SGNify, a model that captures hand pose, facial expression, and body movement from sign language videos. It employs linguistic priors and constraints on 3D hand pose to effectively address the ambiguities in isolated signs.|
|174|2021|the relationship between Two- Hands|
|175|2021|the relationship between Hand-Object|
||2023|HMP: Hand Motion Priors for Pose and Shape Estimation from Video|先用无视频信息的手势数据做手势动作先验。基于先验再做手势识别|手、开源|[link](https://caterpillarstudygroup.github.io/ReadPapers/15.html)|

#### Whole Body

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|176|
|177|
|178|2021|independently running 3D mesh recovery regression for face, hands, and body and subsequently combining the outputs through an integration module|
|179|2021|integrates independent es- timates from the body, face, and hands using the shared shape space of SMPL-X across all body parts|
|180|2022|Accurate 3d hand pose estimation for whole-body 3d human mesh estimation|end-to-end framework for whole-body human mesh recovery named Hand4Whole, which employs joint features for 3D joint rotations to enhance the accuracy of 3D hand predictions|
|181|2023|Pymaf-x: Towards well-aligned full-body model regression from monocular images|to resolve the misalignment issues in regression-based, one-stage human mesh recovery methods by employing a feature pyramid approach and refining the mesh-image alignment parameters.|
|215|
|182|2023|One-stage 3d whole-body mesh recovery with component aware transformer|a simple yet effective component-aware transformer that includes a global body encoder and a lo- cal face/hand decoder instead of separate networks for each part|
|183|

## Template-free human body recovery

# 运动相机场景

## 提取相机轨迹

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|