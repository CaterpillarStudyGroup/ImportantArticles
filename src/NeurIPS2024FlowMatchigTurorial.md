P1     
# Flow Matching Tutorial     

P2     
## Agenda   

[40 mins] **01 Flow Matching Basics**     
[35 mins] **02 Flow Matching Advanced Designs**     
[35 mins] **03 Model Adaptation**     
[30 mins] **04 Generator Matching and Discrete Flows**    
[10 mins] **05 Codebase demo**    

P4     
01 Flow Matching Basics   

P6    
WHAT IS FLOW MATCHING?       
A scalable method to train **flow generative models**.      

HOW DOES IT WORK?      
Train by regressing a **velocity**, sample by following the **velocity**      

P8    
## The Generative Modeling Problem

![](./assets/P8图.PNG)


P10    
## Model

• Continuous-time Markov process       

![](./assets/P10图1.PNG)

![](./assets/P10图2.PNG)


$$
(X_t)_{0\le t\le 1}
$$

$$

$$

P11    

## Marginal probability path

![](./assets/P11图.PNG)


P12   
• For now, we focus on flows…    

![](./assets/P12图.PNG)



P13    
## Flow as a generative model    

![](./assets/P13图.PNG)
