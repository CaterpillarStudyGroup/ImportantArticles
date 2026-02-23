# Summary

- [Introduction](README.md)

# 动画3D管线 - 3DMesh的驱动

- [基于骨骼代理的Mesh的驱动](MeshAnimation/SkeletonProxy/SkeletonProxy.md)
  - [骨骼动作先验](MeshAnimation/SkeletonProxy/MotionPrior.md)
  - [基于离散表示的骨骼动作生成](./MeshAnimation/SkeletonProxy/MotionGenerationDiscreteRepresentation.md)
  - [基于连续表示的骨骼动作生成]()
    - [locomotion](./MeshAnimation/SkeletonProxy/Locomotion.md)
    - [文生动作]()
      - [基于Diffusion的文生动作](./MeshAnimation/SkeletonProxy/MotionGeneration/Text2Motion/DiffusionBasedText2Motion.md)
      - [基于Mamba的文生动作](./MeshAnimation/SkeletonProxy/MotionGeneration/Text2Motion/MambaBasedText2Motion.md)
  - [基于视觉的人类骨骼动作捕捉HPE](./MeshAnimation/SkeletonProxy/HPE_HMR_Summary.md)
  - [facial and expression](CharacterAnimation/HumanFacialAnimation.md)
  - [Human Motion Generation: A Survey](CharacterAnimation/HumanMotionGenerationSummary.md)
- [无代理的Mesh驱动](MeshAnimation/E2E.md)

# 动画3D管线 - Nerf驱动

- [NerfAnimation](NerfAnimation.md)

# 动画3D管线 - 3DGS的驱动

- [3DGS VS. Nerf](./3DGSAnimation/3DGS.md)
- [动态3DGS](./3DGSAnimation/Dynamic.md)
- [静态3DGS](./3DGSAnimation/Static.md)D
  - [基于静态3DGS的4D重建](./3DGSAnimation/4DReconstruction.md)
  - [静态3DGS驱动](./3DGSAnimation/3DGSAnimation.md)
- [A Survey on 3D Gaussian Splatting](3D_Gaussian_Splatting.md)
- [Animal Generation](AnimationGeneration.md)

# 动画2D管线 - 像素的驱动，可控视频生成

- [Introduction](VideoDiffusionModels/Introduction.md)
- [Video Generation](VideoDiffusionModels/VideoGeneration/VideoGeneration.md)
  - [闭源T2V大模型](VideoDiffusionModels/VideoGeneration/Pioneeringearlyworks.md)
  - [开源T2V基模型](VideoDiffusionModels/VideoGeneration/Open-sourcebasemodels.md)
  - [Works Based on T2I 基模型](./VideoDiffusionModels/VideoGeneration/WorksBasedOnT2I.md)
  - [Works Based on T2V 基模型](./VideoDiffusionModels/VideoGeneration/WorksBasedOnT2V.md)
  - [Storyboard](VideoDiffusionModels/VideoGeneration/Storyboard.md)
  - [Long video generation/Storyboard](VideoDiffusionModels/VideoGeneration/Longvideogeneration.md)
  - [Multimodal-guided generation](VideoDiffusionModels/VideoGeneration/Multimodal-guidedgeneration.md)
  - [Human Video Generation](CharacterAnimation/HumanVideoGeneration.md)
- [Video Editing](VideoDiffusionModels/VideoEditing.md)
  - [Tuning-based](VideoDiffusionModels/VideoEditing/Tuning-based.md)
  - [Training-free](VideoDiffusionModels/VideoEditing/Training-free.md)
  - [Controlled Editing](VideoDiffusionModels/VideoEditing/ControlledEditing.md)
  - [3D-Aware](VideoDiffusionModels/VideoEditing/3D-Aware.md)
  - [Other Guidance](VideoDiffusionModels/VideoEditing/OtherGuidance.md)
- [评价指标](./VideoDiffusionModels/EvaluationMetrics.md)

# 动画2D管线 - 2D图形的驱动

- [2D图形驱动](ClipAnimation.md)

# 通用AI技术

## 物理仿真

- [2025 PINN Survey](PhysicsSimulation/PINN.md)
- [2024 Fluid Survey](./PhysicsSimulation/Fluid.md)

## NeurIPS 2024 Flow Matchig Turorial

- [NeurIPS 2024 Flow Matchig Turorial](NeurIPS2024FlowMatchigTurorial/Agenda.md)
   - [Flow Matching Basics](NeurIPS2024FlowMatchigTurorial/FlowMatchingBasics.md)
   - [Flow Matching Advanced Designs](NeurIPS2024FlowMatchigTurorial/FlowMatchingAdvancedDesigns.md)
   - [Model Adaptation](NeurIPS2024FlowMatchigTurorial/ModelAdaptation.md)
   - [Generator Matching and Discrete Flows](NeurIPS2024FlowMatchigTurorial/GeneratorMatchingandDiscreteFlows.md)

## CVPR Tutorial - Denoising Diffusion Models: A Generative Learning Big Bang
- [Introduction](diffusion-tutorial-part/Introduction.md)
- [Fundamentals]()
  - [Denoising Diffusion Probabilistic Models](diffusion-tutorial-part/Fundamentals/DenoisingDiffusionProbabilisticModels.md)
  - [Score-based Generative Modeling with Differential Equations](diffusion-tutorial-part/Fundamentals/Score-basedGenerativeModelingwithDifferentialEquations.md)
  - [Accelerated Sampling](diffusion-tutorial-part/Fundamentals/AcceleratedSampling.md)
  - [Conditional Generation and Guidance](diffusion-tutorial-part/Fundamentals/ConditionalGenerationandGuidance.md)
  - [Summary](./diffusion-tutorial-part/Fundamentals/Summary.md)
- [T2I 基模型](./diffusion-tutorial-part/Architecture.md)
- [Image Applications Based on 基模型]()
  - [图像生成/编辑](diffusion-tutorial-part/ApplicationOnImage/ImageEditing.md)
  - [图像去噪/图像超分/图像补全](diffusion-tutorial-part/ApplicationOnImage/InverseProblems.md)
  - [大图生成](diffusion-tutorial-part/ApplicationOnImage/LargeContents.md)
- [3D Applications Based on Diffusion]()
  - [基于T2I基模型](diffusion-tutorial-part/ApplicationsOn3D/2Ddiffusionmodelsfor3Dgeneration.md)
  - [基于不同视角的3D生成](diffusion-tutorial-part/ApplicationsOn3D/3D.md)
  - [新视角合成](diffusion-tutorial-part/ApplicationsOn3D/Diffusionmodelsforviewsynthesis.md)
  - [3D重建](diffusion-tutorial-part/ApplicationsOn3D/3Dreconstruction.md)
  - [3D编辑](diffusion-tutorial-part/ApplicationsOn3D/Inverseproblems.md)
- [Safety and limitations of diffusion models](diffusion-tutorial-part/ApplicationsOn3D/Safetyandlimitationsofdiffusionmodels.md)
- [Large Multimodal Models Notes on CVPR 2023 Tutorial](LargeMultimodalModelsNotesonCVPR2023Tutorial.md)
- [生成模型](GenerativeModels.md)

# Others

- [数据集](数据集.md)
- [More](More.md)


# Views

- [2025.9.3骨骼动作生成](Views/20250903.md)
- [2025.9.14骨骼动作离散编码](Views/20250914.md)
- [2025.9.20视频可控生成](Views/20250920.md)


  





