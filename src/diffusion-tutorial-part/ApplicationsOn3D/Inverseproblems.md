|ID|Year|Name|Note|Tags|Link|
|---|---|---|---|---|---|
||2023|Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions||Nerf|
||2023|Vox-E: Text-guided Voxel Editing of 3D Objects||Voxel||


P46   
# Instruct NeRF2NeRF

Edit a 3D scene with text instructions   

![](../../assets/D3-46.png)  

Haque et al., <u>"Instruct-NeRF2NeRF: Editing 3D Scenes with Instructions",</u> arXiv 2023      

> &#x2705; 用 Nerf 来描述 3D scene。通过文件条件把原 Nerf，变成另一个 Nerf，从而得到新的 3D scene.    

P47   
## 方法   

**Edit a 3D scene with text instructions**   
 -  Given existing scene, use Instruct Pix2Pix to edit image at different viewpoints.   
 - Continue to train the NeRF and repeat the above process   

![](../../assets/D3-47.png)    

> &#x2705; 首先有一个训好的 Nerf. 对一个特定的场景使用 Instruct Pix 2 pix 在 2D 上编辑训练新的 Werf.    
> &#x2705; 基于 score disllation sampling.    

P48   

With each iteration, the edits become more consistent.    

P49   
# Vox-E: Text-guided Voxel Editing of 3D Objects    

 - Text-guided object editing with SDS
 - Regularize the structure of the new voxel grid.

![](../../assets/D3-49.png)  

Sella et al., <u>"Vox-E: Text-guided Voxel Editing of 3D Objects",</u> arXiv 2023   

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/