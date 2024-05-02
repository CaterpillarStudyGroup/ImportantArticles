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

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
|[123]|2019|
|[124]|2022|
|[125]|2022|
|[126]|2022|
||2023|WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion||单人，移动相机|[link](https://caterpillarstudygroup.github.io/ReadPapers/11.html)

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

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2024|PhysPT: Physics-aware Pretrained Transformer for Estimating Human
Dynamics from Monocular Videos||物理|


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
|167|2023|Jotr: 3d joint con-trastive learning with transformers for occluded human mesh recovery|融合 2D 和 3D 特征，并通过基于 Transformer 的对比学习框架结合对 3D 特征的监督||
|162|2023|Refit: Recurrent fitting network for 3d human recovery|通过反馈-更新循环机制重新投影关键点并完善人体模型||
|4|2023|Co-evolution of pose and mesh for 3d human body estimation from video|引入了一种利用 3D 姿势作为中介的人体mesh恢复的共同进化方法。该方法将过程分为两个不同的阶段：首先，它从视频中估计 3D 人体姿势，随后，根据估计的 3D 姿势并结合时间图像特征对mesh顶点进行回归|开源、单人、视频、mesh|[link](https://caterpillarstudygroup.github.io/ReadPapers/13.html)
|168|2023|Cyclic test-time adaptation on monocular video for 3d human mesh reconstruction|为了弥合训练和测试数据之间的差距，CycleAdapt [168]提出了一种域自适应方法，包括mesh重建网络和运动降噪网络，能够实现更有效的自适应。||

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

### 结合额外的传感器或摄像头

### 仅从视频中提取

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2022|BodySLAM: Joint Camera Localisation, Mapping, and Human Motion Tracking|
||2023|Decoupling Human and Camera Motion from Videos in the Wild|联合优化人体姿势和相机scale，使人体位移与学习的运动模型相匹配|多人|[link](https://caterpillarstudygroup.github.io/ReadPapers/16.html)|
||2024|TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Video|||[link](https://caterpillarstudygroup.github.io/ReadPapers/18.html)|

# Evaluation

## Evaluation metrics

### For pose and shape reconstruction

mean per-joint error (MPJPE),   Procrustes-aligned perjoint error (PA-MPJPE),   
per-vertex error (PVE)

### To evaluate the motion smoothness

acceleration error (ACCEL) against the ground truth acceleration
 
### For human trajectory evaluation, 
 
we slice a sequence into 100-frame segments and evaluate 3D joint error after aligning the first two frames (W-MPJPE100) or the entire segment (WA-MPJPE100) [93].   
evaluate the error of the entire trajectory after aligning the first frame, with root translation error (RTE), root orientation error (ROE), and egocentric root velocity error (ERVE).   

### For camera trajectory evaluation

absolute trajectory error (ATE) [75], which performs Procrustes with scaling to align the estimation with ground truth before computing error.  

### To evaluate the accuracy of our scale estimation

evaluate ATE using our estimated scale (ATE-S) [35].