|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|NeuralLift-360: Lifting An In-the-wild 2D Photo to A 3D Object with 360° Views|SDS + Fine-tuned CLIP text embedding + Depth supervision> &#x2705; 整体上是类似 SDS 的优化方法，再结合其它的损失函数。<br> &#x2705; (1) 渲染不同视角，并对渲染结果用 clip score打分。<br> &#x2705; (2) 监督深度信息。    |![](../../assets/D3-43.png) |
||2023|Zero-1-to-3: Zero-shot One Image to 3D Object|Generate novel view from 1 view and pose, with 2d model. <br> Then, run SJC / SDS-like optimizations with view-conditioned model. <br> &#x2705; (1) 用 2D diffusion 生成多视角。用 SDS 对多视角图像生成3D．   |![](../../assets/D3-44.png) |
||2024|CAT3D|

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/