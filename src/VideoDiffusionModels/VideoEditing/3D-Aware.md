# 3.4 3D-Aware

P243   

![](../../assets/08-243.png) 

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Layered Neural Atlases for Consistent Video Editing| - Decompose a video into a foreground image + a background image <br> - Edit the foreground/background image = edit the video <br> &#x2705; 对背景进行编辑（图片编辑、风格迁移）再传播到不同帧上去。  |![](../../assets/08-244.png) <br> ![](../../assets/08-245.png) |
||2023|VidEdit: Zero-Shot and Spagally Aware Text-Driven Video Edigng | Atlas-based video editing <br> - Decompose a video into a foreground image + a background image <br> - Edit the foreground/background image = edit the video <br> - Use diffusion to edit foreground/background atlas > &#x2705; 前景编辑：<br> (1) 抠出第一帧前景并进行编辑得到 Partial Atlas. <br> &#x2705; (2) Partial Atlas 作为下一帧的 condition 整体上是自回归的。<br> &#x2705; 所有 Partial 合起来得到一个整体。<br> &#x2705; 背景使用深度信息作为 cordition.   | ![](../../assets/08-246.png) |
||2023|Shape-aware Text-driven Layered Video Editing|Atlas-based video editing | ![](../../assets/08-247.png) 
||2023.11|Stablevideo: Text-driven consistency-aware diffusion video editing|&#x2705; 给一个场景的多视角图片，基于 MLP 学习 3D 场景的隐式表达。 |![](../../assets/08-248.png) |  
|115|2023|CoDeF: Content Deformation Fields for Temporally Consistent Video Processing|||[link](https://caterpillarstudygroup.github.io/ReadPapers/115.html)|
||2023|HOSNeRF: Dynamic Human-Object-Scene Neural Radiance Fields from a Single Video|
|116|2023|DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing|||[link](https://caterpillarstudygroup.github.io/ReadPapers/116.html)|

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/