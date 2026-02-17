# Loc
任务：角色动画的实时控制


|ID|Year|Name|解决了什么痛点|主要贡献是什么|Tags|Link|
|---|---|---|---|---|---|---|
||2020.7.8|	Local motion phases for learning multi-contact character movements|① 全局相位模型（如 PFNN）无法适配异步运动；② 时序模型（如 LSTM）响应慢、需大量数据增强；③ 物理基方法可扩展性差，难以学习大规模动作库；④ 生成运动缺乏多样性，无法满足交互需求。|1. 局部运动相位（LMP）LMP 是解决异步多接触运动建模的核心，通过单个骨骼的接触状态自动提取，无需全局对齐<br> 2. 针对 “抽象控制信号→多样化运动” 的映射模糊问题，采用对抗训练的编码器 - 解码器架构 <br> 3. 采用 “门控网络 + 运动预测网络” 的混合专家架构，动态适配不同运动模式|	[link](https://www.pure.ed.ac.uk/ws/files/157671564/Local_Motion_Phases_STARKE_DOA27042020_AFV.pdf)	|
||2018.10.4|Recurrent Transition Networks for Character Locomotion|传统方法（如运动图、高斯过程模型）存在泛化性差、仅适配单一动作类型、运行时计算成本高等问题。<br>基于深度学习的“**从当前状态到目标状态的定向过渡生成**”处于研究空白|1. 改进型 LSTM：传统 LSTM 仅依赖历史状态，RTN 在门控计算中加入未来上下文特征（目标 + 偏移），使生成过程始终朝向目标状态，避免无约束漂移；<br>2. 隐藏状态初始化：摒弃 “零向量初始化” 或 “全局共享初始状态”，通过 MLP 学习输入首帧与最优初始隐藏状态的映射，让 LSTM 从初始阶段就捕捉运动特征，提升生成质量；<br> 3. ResNet 风格解码器：输出当前帧与下一帧的偏移量，而非直接输出姿态，减少生成帧与输入上下文的间隙，提升过渡流畅性。|[link](https://arxiv.org/pdf/1810.02363)|
||2018.6.30|	Mode-adaptive neural networks for quadruped motion control|传统数据驱动方法（运动图、运动匹配）：需存储完整动作数据库，依赖手动分割、标注，搜索过程复杂，实时性差<br>CNN 存在输入输出映射模糊问题，RNN 长期预测易收敛到平均姿态（漂浮），PFNN 虽解决模糊问题，但依赖手动相位标注，无法适配非循环运动。|本文提出模式自适应神经网络（MANN），专为四足动物运动控制设计，核心通过 “门控网络 + 运动预测网络” 的双模块架构，从大规模非结构化动作捕捉（MOCAP）数据中自主学习运动模式，无需手动标注相位或步态标签；门控网络基于脚部关节速度、目标速度等特征动态加权多个专家网络输出，运动预测网络生成平滑连贯的下一帧运动，支持怠速、移动、跳跃、坐姿等多种循环与非循环运动，同时允许用户通过速度、方向、动作变量交互控制；实验证明该模型在运动质量、实时性、内存占用上优于传统数据驱动方法（如运动图）和现有神经网络模型（如 PFNN），填补了四足动物非结构化运动数据高效建模与交互控制的空白。|	[PPT](https://slides.games-cn.org/pdf/GAMES201859%E5%BC%A0%E8%B4%BA.pdf)、[视频](https://v.qq.com/x/page/u0760b6r94p.html)|
|113|2017.7.20| Phase-functioned neural networks for character control|PFNN||[link](https://caterpillarstudygroup.github.io/ReadPapers/113.html)|
||2007.7.29|	SIMBICON: simple biped locomotion control|① 零力矩点（ZMP）方法依赖预计算轨迹，灵活性不足；<br>② 强化学习 / 策略搜索需设计复杂奖励函数，高维状态下难以收敛；<br>③ 数据驱动方法多为运动学建模，缺乏物理适应性。	|“有限状态机 + 全局坐标控制 + 质心反馈” 的极简组合，无需复杂动力学建模，实现实时、鲁棒的物理基双足运动生成|[link](https://www.cs.sfu.ca/~kkyin/papers/Yin_SIG07.pdf)|
