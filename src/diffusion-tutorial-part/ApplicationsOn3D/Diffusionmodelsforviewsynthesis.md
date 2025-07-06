P40   

|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2024b|Unique3D|
||2023|Novel View Synthesis with Diffusion Models|Sample based on stochastic conditions, allowing the use of multiple conditional frames. <br> &#x2705; UNet，2 branch，分别用于原始角度和要生成的角度。<br> &#x2705; 引入 step 2 是为了内容一致性。 <br> &#x2705; frame：坐标系。在不同的坐标系下看到的是不同的视角。 <br> &#x2753; 为什么有两个pose？<br> &#x2705; 每个 frame 的内部由 cross-attention 连接。    | - Condition on a frame and two poses, predict another frame. <br>![](../../assets/D3-40-1.png)  <br> UNet with frame cross-attention <br> ![](../../assets/D3-40-2.png)  |3Dim||
||2024|CAT3D|
||2023|Generative Novel View Synthesis with 3D-Aware Diffusion Models|  - 3D-aware architecture with latent feature field. <br> - Use diffusion model to improve render quality based on structure. <br> &#x2705; (1) 生成 feature field (2) render 其中一个视角 (3) 优化渲染效果 <br> &#x2705; (2) 是 MLP (3) 是 diffusion．| ![](../../assets/D3-41.png)  |GenVS|

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/