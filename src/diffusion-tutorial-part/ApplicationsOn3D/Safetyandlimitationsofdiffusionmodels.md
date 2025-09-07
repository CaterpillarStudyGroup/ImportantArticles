
P73   
## Outline

 - Safety and limitations of diffusion models   

P74   
## Data Memorization in Diffusion Models

 - Due to the likelihood-base objective function, **diffusion models can ”memorize” data**.    
 - And with a higher chance than GANs!   
 - Nevertheless, a lot of “memorized images” are highly-duplicated in the dataset.    

![](../../assets/D3-74.png)  

Carlini et al., <u>"Extracting Training Data from Diffusion Models",</u> arXiv 2023    

P75   
## Erasing Concepts in Diffusion Models   

 - Fine-tune a model to remove unwanted concepts.   
 - From original model, **obtain score via negative CFG**.   
 - **A new model is fine-tuned** from the new score function.    

![](../../assets/D3-75-1.png)  

![](../../assets/D3-75-2.png)  

Gandikota et al., <u>"Erasing Concepts from Diffusion Models",</u> arXiv 2023    

> &#x2705; 考虑到版权等问题。    
> &#x2705; finetune 已有的 text-2-image model．   
> &#x2705; 使用 negative CFG 原有信息不会受到影响。    


# Reference
P77   
## Part I

Ho et al., <u>"Denoising Diffusion Probabilistic Models",</u> NeurIPS 2020     
Kingma et al., <u>"Variational Diffusion Models",</u> arXiv 2021   
Karras et al., <u>"Elucidating the Design Space of Diffusion-Based Generative Models",</u> NeurIPS 2022   
Song et al., <u>"Denoising Diffusion Implicit Models",</u> ICLR 2021   
Jolicoeur-Martineau et al., "Gotta Go Fast When Generating Data with Score-Based Models",</u> arXiv 2021   
Liu et al., <u>"Pseudo Numerical Methods for Diffusion Models on Manifolds",</u> ICLR 2022   
Lu et al., <u>"DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps",</u> NeurIPS 2022   
Lu et al., <u>"DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models",</u> NeurIPS 2022   
Zhang and Chen, <u>"Fast Sampling of Diffusion Models with Exponential Integrator",</u> arXiv 2022   
Zhang et al., <u>"gDDIM: Generalized denoising diffusion implicit models",</u> arXiv 2022   
Zhao et al., <u>"UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models",</u> arXiv 2023    
Shih et al., <u>"Parallel Sampling of Diffusion Models",</u> arxiv 2023   
Chen et al., <u>"A Geometric Perspective on Diffusion Models",</u> arXiv 2023   
Xiao et al., <u>"Tackling the Generative Learning Trilemma with Denoising Diffusion GANs",</u> arXiv 2021   
Salimans and Ho, <u>"Progressive Distillation for Fast Sampling of Diffusion Models",</u> ICLR 2022   
Meng et al., <u>"On Distillation of Guided Diffusion Models",</u> arXiv 2022   
Dockhorn et al., <u>"GENIE: Higher-Order Denoising Diffusion Solvers",</u> NeurIPS 2022   
Watson et al., <u>"Learning Fast Samplers for Diffusion Models by Differentiating Through Sample Quality",</u> ICLR 2022   
Phung et al., <u>"Wavelet Diffusion Models Are Fast and Scalable Image Generators",</u> CVPR 2023   
Dhariwal and Nichol, <u>"Diffusion Models Beat GANs on Image Synthesis",</u> arXiv 2021   
Ho and Salimans, <u>"Classifier-Free Diffusion Guidance",</u> NeurIPS Workshop 2021     
Automatic1111, <u>"Negative Prompt",</u> GitHub   
Hong et al., <u>"Improving Sample Quality of Diffusion Models Using Self-Attention Guidance",</u> arXiv 2022   
Saharia et al., <u>"Image Super-Resolution via Iterative Refinement",</u> arXiv 2021   
Ho et al., <u>"Cascaded Diffusion Models for High Fidelity Image Generation",</u> JMLR 2021   
Sinha et al., <u>"D2C: Diffusion-Denoising Models for Few-shot Conditional Generation",</u> NeurIPS 2021   
Vahdat et al., <u>"Score-based Generative Modeling in Latent Space",</u> arXiv 2021   
Daras et al., <u>"Score-Guided Intermediate Layer Optimization: Fast Langevin Mixing for Inverse Problems",</u> ICML 2022   

P78   
## Part I (cont’d)

Bortoli et al.,<u> "Diffusion Schrödinger Bridge",</u> NeurIPS 2021       
Bortoli et al.,<u> "Riemannian Score-Based Generative Modelling",</u> NeurIPS 2022  
Neklyudov et al., <u>"Action Matching: Learning Stochastic Dynamics from Samples",</u> ICML 2023  
Bansal et al., <u>"Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise",</u> arXiv 2022   
Daras et al., <u>"Soft Diffusion: Score Matching for General Corruptions",</u> TMLR 2023   
Delbracio and Milanfar, <u>"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration",</u> arXiv 2023   
Luo et al., <u>"Image Restoration with Mean-Reverting Stochastic Differential Equations",</u> ICML 2023   

P79    
## Part II
   
Jabri et al., <u>"Scalable Adaptive Computation for Iterative Generation",</u> arXiv 2022        
Li et al., <u>"Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models",</u> NeurIPS 2022   
Avrahami et al., <u>"Blended Diffusion for Text-driven Editing of Natural Images",</u> CVPR 2022          
Sarukkai et al., <u>"Collage Diffusion",</u> arXiv 2023    
Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> ICML 2023      
Kumari et al., <u>"Multi-Concept Customization of Text-to-Image Diffusion",</u> CVPR 2023    
Tewel et al., <u>"Key-Locked Rank One Editing for Text-to-Image Personalization",</u> SIGGRAPH 2023    
Zhao et al., <u>"A Recipe for Watermarking Diffusion Models",</u> arXiv 2023    
Hu et al., <u>"LoRA: Low-Rank Adaptation of Large Language Models",</u> ICLR 2022     
Avrahami et al., <u>"SpaText: Spatio-Textual Representation for Controllable Image Generation",</u> CVPR 2023     
Orgad et al., <u>"Editing Implicit Assumptions in Text-to-Image Diffusion Models",</u> arXiv 2023    
Han et al., <u>"SVDiff: Compact Parameter Space for Diffusion Fine-Tuning",</u> arXiv 2023    
Xie et al., <u>"DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning",</u> arXiv 2023    
Saharia et al., <u>"Palette: Image-to-Image Diffusion Models",</u> SIGGRAPH 2022    
Whang et al., <u>"Deblurring via Stochastic Refinement",</u> CVPR 2022    
Xu et al., <u>"Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models",</u> arXiv 2023    
Saxena et al., <u>"Monocular Depth Estimation using Diffusion Models",</u> arXiv 2023    
Li et al., <u>"Your Diffusion Model is Secretly a Zero-Shot Classifier",</u> arXiv 2023    
Gowal et al., <u>"Improving Robustness using Generated Data",</u> NeurIPS 2021    
Wang et al., <u>"Better Diffusion Models Further Improve Adversarial Training",</u> ICML 2023    

P81   
## Part III   

Jalal et al., <u>"Robust Compressed Sensing MRI with Deep Generative Priors",</u> NeurIPS 2021  
Song et al., <u>"Solving Inverse Problems in Medical Imaging with Score-Based Generative Models",</u> ICLR 2022   
Kawar et al., <u>"Denoising Diffusion Restoration Models",</u> NeurIPS 2022   
Chung et al., <u>"Improving Diffusion Models for Inverse Problems using Manifold Constraints",</u> NeurIPS 2022   
Ryu and Ye, <u>"Pyramidal Denoising Diffusion Probabilistic Models",</u> arXiv 2022   
Chung et al., <u>"Diffusion Posterior Sampling for General Noisy Inverse Problems",</u> arXiv 2022   
Feng et al., <u>"Score-Based Diffusion Models as Principled Priors for Inverse Imaging",</u> arXiv 2023   
Song et al., <u>"Pseudoinverse-Guided Diffusion Models for Inverse Problems",</u> ICLR 2023   
Mardani et al., <u>"A Variational Perspective on Solving Inverse Problems with Diffusion Models",</u> arXiv 2023   
Delbracio and Milanfar, <u>"Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration",</u> arxiv 2023   
Stevens et al., <u>"Removing Structured Noise with Diffusion Models",</u> arxiv 2023   
Wang et al., <u>"Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model",</u> ICLR 2023   
Zhou et al., <u>"3D Shape Generation and Completion through Point-Voxel Diffusion",</u> ICCV 2021   
Zeng et al., <u>"LION: Latent Point Diffusion Models for 3D Shape Generation",</u> NeurIPS 2022   
Nichol et al., <u>"Point-E: A System for Generating 3D Point Clouds from Complex Prompts",</u> arXiv 2022   
Chou et al., <u>"DiffusionSDF: Conditional Generative Modeling of Signed Distance Functions",</u> arXiv 2022   
Cheng et al., <u>"SDFusion: Multimodal 3D Shape Completion, Reconstruction, and Generation",</u> arXiv 2022   
Hui et al., <u>"Neural Wavelet-domain Diffusion for 3D Shape Generation",</u> arXiv 2022   
Shue et al., <u>"3D Neural Field Generation using Triplane Diffusion",</u> arXiv 2022   
Yang et al., <u>"Learning a Diffusion Prior for NeRFs",</u> ICLR Workshop 2023   
Jun and Nichol, <u>"Shap-E: Generating Conditional 3D Implicit Functions",</u> arXiv 2023     
Metzer et al., <u>"Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures",</u> arXiv 2022   
Hong et al., <u>"Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation",</u> CVPR Workshop 2023   
Watson et al., <u>"Novel View Synthesis with Diffusion Models",</u> arXiv 2022   
Chan et al., <u>"Generative Novel View Synthesis with 3D-Aware Diffusion Models",</u> arXiv 2023   
Zhou and Tulsiani, <u>"SparseFusion: Distilling View-conditioned Diffusion for 3D Reconstruction",</u> arXiv 2022   

P82   
## Part III (cont’d)

Seo et al., <u>"DITTO-NeRF: Diffusion-based Iterative Text To Omni-directional 3D Model",</u> arXiv 2023   
Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023   
Sella et al., <u>"Vox-E: Text-guided Voxel Editing of 3D Objects",</u> arXiv 2023   
Harvey et al., <u>"Flexible Diffusion Modeling of Long Videos",</u> arXiv 2022    
Voleti et al., <u>"MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation",</u> NeurIPS 2022     
Mei and Patel, <u>"VIDM: Video Implicit Diffusion Models",</u> arXiv 2022        
Wang et al., <u>"Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models",</u> arXiv 2023      
Jiménez, <u>"Mixture of Diffusers for scene composition and high resolution image generation",</u> arXiv 2023    
Bar-Tal et al., <u>"MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",</u> arXiv 2023    
Zhang et al., <u>"DiffCollage: Parallel Generation of Large Content with Diffusion Models",</u> CVPR 2023       
Du et al., <u>"Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model",</u> CVPR 2023    
Shafir et al., <u>"Human Motion Diffusion as a Generative Prior",</u> arXiv 2023    
Somepalli et al., <u>"Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models",</u> CVPR 2023    
Carlini et al., <u>"Extracting Training Data from Diffusion Models",</u> arXiv 2023    
Gandikota et al., <u>"Erasing Concepts from Diffusion Models",</u> arXiv 2023    
Kumari et al., <u>"Ablating Concepts in Text-to-Image Diffusion Models",</u> arXiv 2023    
Somepalli et al., <u>"Understanding and Mitigating Copying in Diffusion Models",</u> arXiv 2023    



---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/