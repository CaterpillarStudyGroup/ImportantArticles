```mermaid
mindmap
基于diffusion的动作条件的动作生成
    相同点
        生成方式：非自回归
        表示方式：连续表示
        生成模型：DDPM/DDIM
    动作条件
        轨迹
        关键帧动作（锚点姿态）
        特定关节的轨迹
        历史动作
    条件注入方式
        concat
        Control Net
        Overwrite
        推断时的去噪引导
    按要解决的问题分
        与控制信号的匹配
        不同控制信号的解耦
```

## 轨迹与姿态的解耦

|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
|169|2025.5.27|IKMo: Image-Keyframed Motion Generation with Trajectory-Pose Conditioned Motion Diffusion Model|现有基于轨迹与姿态输入的人体运动生成方法通常对这两种模态进行全局处理，导致输出结果次优|其核心在于**解耦轨迹与姿态输入**  ||
|171|2025.4.23|PMG: Progressive Motion Generation via Sparse Anchor Postures Curriculum Learning|轨迹引导与稀疏锚点动作解耦控制|**更高控制精度和更精细的运动生成** ||

## 与控制信号的匹配

|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
|168|2025.5.28|UniMoGen: Universal Motion Generation|**骨架无关**的动作生成  |UNet Based，风格与轨迹控制||
|85|2024|OmniControl: Control Any Joint at Any Time for Human Motion Generation|1. 使用ControlNet方式引入控制信号<br>2. 使用推断时损失注入方式进一步实现空间约束。|based on MDM、GMD，精确控制，ControlNet|[link](https://caterpillarstudygroup.github.io/ReadPapers/85.html)|
||2024.5|Flexible motion in-betweening with diffusion models|生成能够合理插值用户提供的关键帧约束的运动序列|支持任意稠密或稀疏关键帧布局及部分关键帧约束，同时生成与给定关键帧保持连贯性的高质量多样化运动|based on GMD |[link](https://arxiv.org/pdf/2405.11126)|
|86|2023|Guided Motion Diffusion for Controllable Human Motion Synthesis|将空间约束融入运动生成过程, 通过two-stage pipeline解决控制信号稀疏导致控制能力不足的问题。<br>第一阶段通过提升root投影轨迹loss强化轨迹控制，通过去噪函数实现稀疏轨迹->稠密轨迹的方法，从而生成稠密轨迹。<br>第二阶段使用稠密信号引导生成|GMD，轨迹控制|[link](https://caterpillarstudygroup.github.io/ReadPapers/86.html)|