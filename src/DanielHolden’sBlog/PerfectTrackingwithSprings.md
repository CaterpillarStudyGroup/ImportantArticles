转载出处：https://www.daniel-holden.com/page/perfect-tracking-springs


> &#x1F4A1; 本文主题：
> 1. “tracks the acceleration, velocity, and position of the input animation in different proportions”能够得到比较不错的tracking效果。  
> 2. 直接的tracking在timestep改变时难以自适应，可以借鉴弹簧系统`spring_damper_exact_stiffness_damping`来解决。   
> 3. 弹簧系统和tracking在公式上有相通之处，通过公式推导计算出弹簧系统的关键系数，可以使弹簧系统达到与tracking相同的效果。  

# Perfect Tracking with Springs

Created on Jan. 25, 2023, 3:44 p.m.

If you have some animation data with discontinuities you want to remove, you might have considered tracking that animation data with some kind of spring:

> &#x2705; 为什么弹簧系统可以解决不连续性问题？因为弹簧系统中会考虑质量和惯性，因此运动（位置、速度、加速度。。。。等）都无法瞬间改变。  
> &#x2705; 但是实际上animation data常常是离散数据，每一帧的运动变化都是通过差分算出来的，所以无法区分某一帧的运动数值变化到底是突变（不连续）还是只是运动变化过快（连续）。所以只能设置一个阈值，超过阈值的变化当作是突变。  

This does a good job at removing the discontinuities - but the spring needs to be very stiff - meaning the discontinuities get aggressively blended out - and even worse - it never quite reproduces exactly the input signal in places where nothing is wrong.

> &#x2705; 弹簧非常stiff意味着什么呢？弹簧系数非常大，在这种情况下难以产生很快的运动变化。  
> &#x2705; 为什么弹簧必须非常stiff呢？其实跟阈值限制是一个目的，就是防止物体运动变化太快。只不过阈值限制是非常暴力的一刀切，有可能产生高阶的不连续。而弹簧能保证每一阶的连续性。  
> &#x1F50E; 使用弹簧系统来tracking的效果：
https://www.daniel-holden.com/media/uploads/springs/tracking_bad.m4v  
> &#x2705; 可以看出弹簧系统tracking的这样一些问题：（1）运动会有滞后 （2）跳变平滑之后难以快速跟上原始信号 （3）较快的运动变化会丢失。
> &#x2705; 同时，弹簧系统的stiffness也比较难调整，stiffness过大，（3）会变严重。stiffness过小，（1）和（2）会变严重。  

For this reason, I've always tried to avoid going down this route whenever I could, and to me, inertialization always seemed like the "correct" way of dealing with these kind of discontinuities if you knew where they occurred.

> &#x1F50E; inertialization：惯性差值

However, a few years ago, my old colleague Jack Potter showed me a kind of spring that can filter animation in exactly the way we want: it can remove discontinuities smoothly while still perfectly tracking the original signal when nothing is wrong.

> &#x2705; 不只去跳变，任何以内容优化为目的的算法，都是这样的目标：在去除不好的同时尽可能地保留好的。  

The trick is to make a spring that **tracks the acceleration, velocity, and position of the input animation in different proportions**.

In code it looks something like this:

```c++
void tracking_spring_update(
    float& x,
    float& v, // current velocity
    float x_goal, // the position of the input animation
    float v_goal, // the velocity of the input animation itself
    float a_goal, // the acceleration of the input animation
    float x_gain, // gain to control the strength, e.g. 0.01
    float v_gain, // e.g. 0.2
    float a_gain, // e.g. 1
    float dt) // e.g. 60 frames per second
{
    // a对v的影响方式是a*dt
    v = lerp(v, v + a_goal * dt, a_gain);
    v = lerp(v, v_goal, v_gain);
    // x对v的影响方式是dx/dt
    v = lerp(v, (x_goal - x) / dt, x_gain);
    x = x + dt * v;
}
```

> &#x2705; `xxx_goal`代表原始运动（红线）的状态。`x`和`v`代表tracking moition（蓝线）的状态。  

First, we blend the current velocity with the current velocity plus the acceleration of the input animation multiplied by the dt. Then, we blend this with the velocity of the input animation itself. Finally, we blend this with a velocity that pulls us toward the position of the input animation.

> &#x2705; 根据代码计算出每一项对下一帧速度的影响度：  

||||
|---|---|---|
|当前速度|v|`a_gain * v_gain * x_gain`|
|原始轨迹的加速度|`a_goal * dt`|`(1-a_gain) * v_gain * x_gain)`|
|原始轨迹的速度|`v_goal`|`(1-v_gain * x_gain)`|
|原始轨迹的位置|`(x_goal - x)/dt`|`(1-x_gain)`|

For each of these blends we can use a gain to control the strength. For perfect tracking, we will want to set the acceleration gain to 1, the velocity gain to something like 0.2, and the position gain to something quite small such as 0.01 (when running at 60 frames per second).

> &#x2705; 从建议的参数可以看出，tracking motion主要受原始轨迹的位置变化的影响，但不太关心原始轨迹的加速度。  

This spring will need to be fed the acceleration, velocity, and position of the input signal. The easiest way to get these is by finite difference:

```c++
float tracking_target_acceleration(
    float x_next,
    float x_curr,
    float x_prev,
    float dt)
{
    return (((x_next - x_curr) / dt) - ((x_curr - x_prev) / dt)) / dt;
}

float tracking_target_velocity(
    float x_next,
    float x_curr,
    float dt)
{
    // 后向差分
    return (x_next - x_curr) / dt;
}
```

If we know there is a discontinuity in this signal, we know the values computed by this finite difference will not be good (at least the acceleration and velocity targets wont be) - so we just ignore them for those time steps and only blend the velocity with values we know are good:

> &#x2705; 当原始轨迹的v和a不正确时，就不要受它们的影响了。可是怎么知道v和a是否正确呢？  

```c++
void tracking_spring_update_no_acceleration(
    float& x,
    float& v,
    float x_goal,
    float v_goal,
    float x_gain,
    float v_gain,
    float dt)
{
    v = lerp(v, v_goal, v_gain);
    v = lerp(v, (x_goal - x) / dt, x_gain);
    x = x + dt * v;
}

void tracking_spring_update_no_velocity_acceleration(
    float& x,
    float& v,
    float x_goal,
    float x_gain,
    float dt)
{
    v = lerp(v, (x_goal - x) / dt, x_gain);
    x = x + dt * v;
}
```

This allows the spring to **ignore** the discontinuity and naturally converge back onto the input stream, tracking it perfectly:

> &#x1F50E; https://www.daniel-holden.com/media/uploads/springs/tracking_good.m4v

If we don't know where the discontinuities are in the input signal we have two options - we can try to detect them and ignore them based on some kind of **heuristic** - or we can **clip** whatever we get to some maximum and minimum velocity and acceleration. Clipping can still sometimes give us a nasty jump when discontinuities occur but works okay given the imperfect situation we might be in:

> &#x2705; 因此，当发现原始运动的v和a不合理时，有两种处理方法：（1）ignore，只参考原始动作的位置信息。（2）clip，截断。  

> &#x1F50E; https://www.daniel-holden.com/media/uploads/springs/tracking_clamp.m4v

This can be useful when our input signal may jump around in hard to specify ways or when it is coming from some black box we can't control.

One important thing to note here is that this code is not in any way robust to varying timesteps! The use of these gains will give us very different results if we tick at different rates:

> &#x2705; timestep比较大的时候，似乎要花很久才能在跳变之后追上原始信号。  

> &#x1F50E; https://www.daniel-holden.com/media/uploads/springs/tracking_gain_dt.m4v

We can improve things somewhat by switching `lerp` to `damper_exact` and using half-lives to control the blending (in this case I use 0.0 for the acceleration halflife, 0.05 for the velocity halflife and 1.0 for the position halflife).

> &#x2753; `damper_exact`是什么？没有找到相关材料。  

```c++
void tracking_spring_update_improved(
    float& x,
    float& v,
    float x_goal,
    float v_goal,
    float a_goal,
    float x_halflife,
    float v_halflife,
    float a_halflife,
    float dt)
{
    v = damper_exact(v, v + a_goal * dt, a_halflife, dt);
    v = damper_exact(v, v_goal, v_halflife, dt);
    v = damper_exact(v, (x_goal - x) / dt, x_halflife, dt);
    x = x + dt * v;
}
```

This version is certainly far from dt-invariant either, but feels a bit better than before to me.

> &#x2753; dt-invariant version是指什么？  
> &#x1F50E; https://www.daniel-holden.com/media/uploads/springs/tracking_damper_dt.m4v

Although basically stable, I would warn against ticking this spring at a variable timestep. Other than that I think it is pretty neat.

If anyone wants to go through the maths to try and derive an exact, dt-invariant version of this spring that would be awesome - or if anyone knows if this spring exists under a different name in engineering/control theory I would love to know!

That's all for now.

---

## Appendix: Exact Tracking Spring

So after a little discussion on Twitter it seemed that (although there were some good suggestions), no one could identify exactly what kind of spring this was. A couple of the replies however did seed some ideas in my head and I decided to try and see if I could derive a dt-invariant version of this spring by playing around with the equations. Spoiler: it turns out this spring is equivalent to our normal [spring-damper](https://www.daniel-holden.com/page/spring-roll-call#springdamper%22) with specifically tuned stiffness, damping, goal position, and goal velocity.

To see how this works, we need to break down the tracking spring velocity equation into its different parts.

First, we have the continuation of the existing velocity, which will end up scaled by some amount depending on our gains. We will denote this scaling using some variable \\( \theta_0 \\):

\begin{align*} v_{t} &= \theta_0 \cdot v_t + ... \end{align*}

Next, we have the contribution from the acceleration of the signal being tracked \\( g_a \\). This is multiplied by the delta time \\( dt \\), and again scaled by some amount depending on the gains. Similarly to before, we'll denote this as \\( \theta_1 \\):

\begin{align*} v_{t} &= \theta_0 \cdot v_t + dt \cdot \theta_1 \cdot g_a + ... \end{align*}

Then, we have the contribution from the velocity of the signal being tracked \\( g_v \\), which we will multiply by the delta time \\( dt \\) and our third scaling variable \\( \theta_2 \\):

\begin{align*} v_{t} &= \theta_0 \cdot v_t + dt \cdot \theta_1 \cdot g_a + dt \cdot \theta_2 \cdot g_v + ... \end{align*}

Finally, we have the term that pulls the spring toward the position of the signal being tracked \\( g_x \\). Like the others, this we will scale by the delta time \\( dt \\) and another variable \\( \theta_3 \\):

\begin{align*} v_{t} &= \theta_0 \cdot v_t + dt \cdot \theta_1 \cdot g_a + dt \cdot \theta_2 \cdot g_v + dt \cdot \theta_3 \cdot (g_x - x_{t}) \end{align*}

> &#x2705; 最后一项应该除以dt，而不是乘
> &#x2705; 参考上面的表格  

||||
|---|---|---|
|当前速度|v|`theta_0 = a_gain * v_gain * x_gain`|
|原始轨迹的加速度|`a_goal * dt`|`theta_1 = (1-a_gain) * v_gain * x_gain)`|
|原始轨迹的速度|`v_goal`|`theta_2 = (1-v_gain * x_gain)`|
|原始轨迹的位置|`(x_goal - x)/dt`|`theta_3 = (1-x_gain)`|

We can think of this as simply another formulation of our tracking spring using a different set of variables. What we need to do now is find a way to compute these values \\( \theta_0 \\), \\( \theta_1 \\), \\( \theta_2 \\), \\( \theta_3 \\) from our gains. The way we are going to do this is to kind of work backwards to find assignments of these that reflect the behavior we expect from the lerp formulation.

First, what are the values of \\( \theta_0 \\) and \\( \theta_1 \\) according to our acceleration gain \\( \alpha_a \\)? This one is simple - the acceleration gain \\( \alpha_a \\) doesn't affect the scale of the existing velocity, meanwhile when \\( \alpha_a = 1 \\) we expect to have the full contribution from \\( g_a \\) - effectively \\( \theta_1 = \alpha_a \\) in this case:

\begin{align*} \theta_0 &= 1 \\\\ \theta_1 &= \alpha_a \\ \end{align*}

> &#x2753; 这里似乎跟我理解的不太一样，可能要先搞清楚`damper_exact`。  

Next, what happens when we introduce the velocity gain \\( \alpha_v \\)? As well as scaling our velocity target \\( g_v \\) up such that \\( dt \cdot \theta_2 = 1 \\) when \\( \alpha_v = 1 \\), it will also scale down the values of \\( \theta_0 \\) and \\( \theta_1 \\) in equal proportion.

\begin{align*} \theta_0 &= 1 - \alpha_v \\\\ \theta_1 &= \alpha_a \cdot (1 - \alpha_v) \\\\ \theta_2 &= \frac{\alpha_v}{dt} \end{align*}

Finally, we need to account for our position gain \\( \alpha_x \\). When \\( \alpha_x = 1 \\) we expect the result to be pulled directly onto the tracking target \\( g_x \\), which means \\( \alpha_3 = \tfrac{\alpha_x}{dt^2} \\). Also, just like the velocity gain, the position gain will scale down the other terms as it grows larger:

\begin{align*} \theta_0 &= (1 - \alpha_v) \cdot (1 - \alpha_x) \\\\ \theta_1 &= \alpha_a \cdot (1 - \alpha_v) \cdot (1 - \alpha_x) \\\\ \theta_2 &= \frac{\alpha_v \cdot (1 - \alpha_x)}{dt} \\\\ \theta_3 &= \frac{\alpha_x}{dt^2} \end{align*}

But why is that useful? Well our goal now is going to be to rearrange our equation with \\( \theta_0 \\), \\( \theta_1 \\), \\( \theta_2 \\) and \\( \theta_3 \\) to put it in a similar form to our spring-damper velocity equation. If we can find a matching formulation then we can use the time-invariant derivation we've already worked out.

> &#x2753; spring-damper velocity equation

First let's re-arrange our spring-damper velocity equation a bit, where \\( s = stiffness \\), \\( d = damping \\), \\( g \\) is our goal position, and \\( q \\) is our goal velocity:

> &#x2705; stiffness: 让弹簧快速回到目标位置  
> &#x2705; damping：让速度变化不要太快  

\begin{align*} v_t = v_t + dt \cdot s \cdot (g - x_t) + dt \cdot d \cdot (q - v_t) \end{align*}

First we'll multiply out the right-most term:

\begin{align*} v_t = v_t + dt \cdot s \cdot (g - x_t) + dt \cdot d \cdot q - dt \cdot d \cdot v_t \end{align*}

Then, we can re-factorize to put the single velocity term along with the rightmost term to get this:

\begin{align*} v_t = dt \cdot s \cdot (g - x_t) + dt \cdot d \cdot q + (1 - dt \cdot d) \cdot v_t \end{align*}

> &#x2705; 重新组合为当前速度项、目标速度项、目标位置项、目标加速度项（这一项没有）  

Now, let's take our other equation...

\begin{align*} v_{t} &= \theta_0 \cdot v_t + dt \cdot \theta_1 \cdot g_a + dt \cdot \theta_2 \cdot g_v + dt \cdot \theta_3 \cdot (g_x - x_{t}) \end{align*}

... and factorize out the terms containing \\( \theta_1 \\) and \\( \theta_2 \\) so they share the same \\( dt \\):

\begin{align*} v_{t} &= \theta_0 \cdot v_t + dt \cdot \theta_1 \cdot g_a + dt \cdot \theta_2 \cdot g_v + dt \cdot \theta_3 \cdot (g_x - x_{t}) \\\\ v_{t} &= dt \cdot \theta_3 \cdot (g_x - x_{t}) + dt \cdot (\theta_1 \cdot g_a + \theta_2 \cdot g_v) + \theta_0 \cdot v_t \end{align*}

Aha! If we look carefully we can see some matches. For the first term we have \\( s = \theta_3 \\), and \\( g = g_x \\). For the second term we have \\( d \cdot q = \theta_1 \cdot g_a + \theta_2 \cdot g_v \\). And for the third term we have \\( 1 - dt \cdot d = \theta_0 \\). While the first two equations give us direct solutions, the second two will require a little re-arranging as we need to solve for \\( d \\) and \\( q \\).

We can solve for \\( d \\) first:

\begin{align*} 1 - dt \cdot d &= \theta_0 \\\\ dt \cdot d &= 1 - \theta_0 \\\\ d &= \frac{1 - \theta_0}{dt} \end{align*}

Which we can then use to solve for \\( q \\)

\begin{align*} d \cdot q &= \theta_1 \cdot g_a + \theta_2 \cdot g_v \\\\ q &= \frac{\theta_1 \cdot g_a + \theta_2 \cdot g_v}{d} \end{align*}

And with that we have our spring-damper parameters \\( s \\), \\( d \\), \\( g \\), \\( q \\), computed from our tracking spring parameters \\( \theta_0 \\), \\( \theta_1 \\), \\( \theta_2 \\) and \\( \theta_3 \\):

\begin{align*} s &= \theta_3 \\\\ d &= \frac{1 - \theta_0}{dt} \\\\ g &= g_x \\\\ q &= \frac{\theta_1 \cdot g_a + \theta_2 \cdot g_v}{d} \\\\ \end{align*}

Now all that remains is to plug these parameters into the exact spring damper equation we already derived and we are done. Here it is implemented in C++, where gain_dt is the timestep which we are emulating running our gains at:

```c++
void tracking_spring_update_exact(
    float& x,
    float& v,
    float x_goal,
    float v_goal,
    float a_goal,
    float x_gain,
    float v_gain,
    float a_gain,
    float dt,
    float gain_dt)
{
    float t0 = (1.0f - v_gain) * (1.0f - x_gain);
    float t1 = a_gain * (1.0f - v_gain) * (1.0f - x_gain);
    float t2 = (v_gain * (1.0f - x_gain)) / gain_dt;
    float t3 = x_gain / (gain_dt*gain_dt);
    
    float stiffness = t3;
    float damping = (1.0f - t0) / gain_dt;
    float spring_x_goal = x_goal;
    float spring_v_goal = (t2*v_goal + t1*a_goal) / ((1.0f - t0) / gain_dt);
    
    spring_damper_exact_stiffness_damping(
      x, 
      v, 
      spring_x_goal,
      spring_v_goal,
      stiffness,
      damping,
      dt);
}
```

And this is what it looks like in action.

> &#x1F50E; https://www.daniel-holden.com/media/uploads/springs/tracking_exact.m4v

In terms of dt-invariance definitely much better than what we had before!

However, one thing you've probably noticed about this version is that it **no longer tracks the input signal pixel perfectly**. This is because the call to `spring_damper_exact` does not perfectly emulate what you would get evaluating the spring-damper with discrete ticks at a rate of `gain_dt` - which is what we are assuming when we compute our spring parameters. I think if we were to set `gain_dt` to a smaller value and tweak the gains we would get better tracking. Alternatively, we could switch to the normal integration based method when running at the framerate specified by `gain_dt`.

> &#x2753; the spring-damper with discrete ticks 这句话怎么理解？  
> &#x2753; 什么是normal integration based method？  

Another thing to note is that this derivation assumes that the target position, velocity and acceleration coming from the input signal are fixed over the full timestep. This means that if you make the \\( dt \\) very large and your signal becomes very coarse the behavior of this spring will be different to what you would get with smaller timesteps. In other words, this dt-invariant derivation becomes less and less effective if the tracking signal is changing quickly.

> &#x2757; 这种推导假设来自输入信号的目标位置、速度和加速度在整个时间步长上是固定的。   
> &#x2705; 因此timestep不能太大。  

Nonetheless, if you do have a timestep that varies, I think this version is still is far, far better than relying on the version we had before.

The other variations we need where **we don't have acceleration or velocity targets** have pretty similar derivations so I wont go through the maths, but in C++ they can be implemented as follows:

> &#x2705; 没有`a_goal`和`v_goal`的情况下，  
> 第一步：\\(\theta_1\\)和\\(\theta_2\\)就不存在了，重新推导出\\(\theta_0\\)和\\(\theta_3\\)  
> 第二步：需要\\(\theta_1\\)和\\(\theta_2\\)的地方用0代替。  

```c++
void tracking_spring_update_no_acceleration_exact(
    float& x,
    float& v,
    float x_goal,
    float v_goal,
    float x_gain,
    float v_gain,
    float dt,
    float gain_dt)
{
    float t0 = (1.0f - v_gain) * (1.0f - x_gain);
    float t2 = (v_gain * (1.0f - x_gain)) / gain_dt;
    float t3 = x_gain / (gain_dt*gain_dt);
    
    float stiffness = t3;
    float damping = (1.0f - t0) / gain_dt;
    float spring_x_goal = x_goal;
    float spring_v_goal = t2*v_goal / ((1.0f - t0) / gain_dt);

    spring_damper_exact_stiffness_damping(
      x, 
      v, 
      spring_x_goal,
      spring_v_goal,
      stiffness,
      damping,
      dt);
}

void tracking_spring_update_no_velocity_acceleration_exact(
    float& x,
    float& v,
    float x_goal,
    float x_gain,
    float dt,
    float gain_dt)
{
    float t0 = 1.0f - x_gain;
    float t3 = x_gain / (gain_dt*gain_dt);
    
    float stiffness = t3;
    float damping = (1.0f - t0) / gain_dt;
    float spring_x_goal = x_goal;
    float spring_v_goal = 0.0f;
  
    spring_damper_exact_stiffness_damping(
      x, 
      v, 
      spring_x_goal,
      spring_v_goal,
      stiffness,
      damping,
      dt);
}
```

And that is about it! All the code for this article can be found on github. I hope you find it useful.

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/