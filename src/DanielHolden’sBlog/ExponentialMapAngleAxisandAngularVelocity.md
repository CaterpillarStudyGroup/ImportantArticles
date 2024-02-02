转载出处：https://www.daniel-holden.com/page/exponential-map-angle-axis-angular-velocity

# Exponential Map, Angle Axis, and Angular Velocity  

Created on July 10, 2021, 5:16 p.m.

If you've used rotations when programming graphics, physics, or video games, there are three concepts you've probably seen mentioned, but may not have understood exactly what the difference was between them all:

  - The exponential map   
  - The angle-axis representation   
  - The angular velocity   

These three concepts are all closely related, so to make sure we're all on the same page, this is exactly how I'm defining them:

  - **The exponential map**: A specific function which takes a 3D vector and produces a rotation from it.    
  - **The angle-axis representation**: A representation of rotation consisting of an axis, and an angle of rotation around that axis.   
  - **The angular velocity**: The angular rate of change of a rotation with respect to time.  

But before we go into more detail there are a few more concepts which we need to define:

The first, which does not seem to have a standard name (sometimes I've heard it called the "helical" representation of a rotation), is to convert a rotation into angle-axis, take the axis, and scale it by the angle of the rotation. The result is a new 3D vector which I like to call the **scaled-angle-axis representation**.

> &#x2753; scaled angle axis与angle axis是什么区别？scale it by the angle是什么意思？  

The second is simply the inverse of the exponential map - a function which takes a rotation in some form (e.g. a quaternion, or rotation matrix), and converts it into the corresponding exponentially mapped 3D vector. Usually this is simply called the **log** of a rotation.

> &#x2753; 和exponential有什么关系？

The confusion around all these three concepts comes down to the fact that the exponential map, the angle-axis representation, and the angular velocity all do more or less the same thing... *sort of*:

  - Taking the "log" of a rotation to get the 3D exponentially mapped version is exactly the same as computing the scaled-angle-axis representation and then dividing it by two.   
  - In other words, the "log" computes the "axis" of a rotation, scaled by the "half angle" of a rotation.   
  - Or putting it another way, if we want to compute the angle and axis of a rotation we can take its "log", and the direction of the result will represent the "axis", while the magnitude will represent half the "angle" of rotation around that axis.   

> &#x2705; angle, axis = log(vector)  
> scaled_angle_axis = half(angle) * axis

The angular velocity is related in a slightly different way: in essence it's the difference between two rotations, stored in the "scaled-angle-axis" representation, and scaled by some "dt":   

  - If we have two rotations, and we multiply one by the inverse of the other, convert the result to scaled-angle-axis, and divide by the "dt" we have exactly the "angular velocity" between these rotations - just as if we were computing the velocity via finite difference.   

> &#x2705; 求两个rotation之间的变化速度：   
> diff = r1 * inverse(r2)  
> saa = scaled_angle_axis(diff)  
> av = saa / dt

  - If you have a set of angular velocities you want to integrate, you simply multiply them by the "dt", convert them all back from the scaled-angle-axis format to whatever rotation format you are using, and multiply all those rotations together.   

> &#x2705; 已知一个角度和一个角速度，求另一个角度：  
> saa = av * dt  
> r2 = saa * r1

Maybe things will be clearer with a bit of code...   

--- 

Because all of these concepts are so interlinked, when it comes down to actually implementing them, it seems natural to implement them in terms of each other, and probably the first approach most people take is to define everything in terms of angle-axis conversion.

But it can actually be beneficial to think more carefully about the differences between all of them. Consider the identity rotation: the angle is clearly zero, but in angle-axis the axis could be any axis. This means you need to pick an arbitrary axis for when the rotation is close to the identity and this produces a weird edge case: converting from angle-axis and back again can give you a different result to what you started with.  

> &#x2705; identity rotation 在轴角表示法中是不唯一的。  

Similarly, converting from angle-axis to another rotation format is often useful for getting rotations from user input or other controls, but converting from some rotation format to angle-axis is rarely required for any other reason than computing angular velocities, scaled-angle-axis, or the "log" function.   

> &#x2705; 轴角表示最大的作用是计算角速度

In fact if we understand the concepts well, all these functions can have relatively simple implementations which make exactly how they work clear too.   

Personally, I think it's easiest to start with `log` and `exp` functions for quaternions. If we take a quick look at the [wikipedia page](https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions) and transcribe the equations into code, here is what we might come up with:  

```c++
quat quat_exp_naive(vec3 v)
{
    float halfangle = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    float c = cosf(halfangle);
    float s = sinf(halfangle) / halfangle;
    return quat(c, s * v.x, s * v.y, s * v.z);
}
```

And...

```c++
vec3 quat_log_naive(quat q)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
    float halfangle = acosf(q.w);
    return halfangle * (vec3(q.x, q.y, q.z) / length);
}
```

But this has some issues. Can you spot them? Well in `quat_log_naive` we can get a division by zero when `length` is zero - which will definitely be the case if we have any quaternion close to the identity quaternion. While in `quat_exp_naive` we have a division by zero when `halfangle` is close to zero... which again will be the case when we have the zero vector (i.e. identity rotation) as input.   

And with that aside, if you actually try these functions you'll notice they still occasionally produce `NaNs`. This is because **the `w` component of the quaternion can often be slightly smaller or larger than `-1` or `1`** due to the floating point error that accumulates when multiplying quaternions together - which **produces `NaN`** when we pass it to the `acosf` function.   

One way to fix these issues is to add some checks, and use an approximation when the rotation is very close to the identity:

```c++
quat quat_exp(vec3 v, float eps=1e-8f)
{
    float halfangle = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	
    if (halfangle < eps)
    {
        return quat_normalize(quat(1.0f, v.x, v.y, v.z));
    }
    else
    {
        float c = cosf(halfangle);
        float s = sinf(halfangle) / halfangle;
        return quat(c, s * v.x, s * v.y, s * v.z);
    }
}
```

And...

```c++
vec3 quat_log(quat q, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
	
    if (length < eps)
    {
        return vec3(q.x, q.y, q.z);
    }
    else
    {
        float halfangle = acosf(clampf(q.w, -1.0f, 1.0f));
        return halfangle * (vec3(q.x, q.y, q.z) / length);
    }
}
```

Where...

```c++
float clampf(float x, float min, float max)
{
    return x > max ? max : x < min ? min : x;
}
```

This also tells us something interesting: that when our rotation is close to the identity rotation both the `exp` and `log` functions boil down to what is essentially just taking the imaginary components of the quaternion. We can create specific versions of these functions for when we need fast approximations:

```c++
quat quat_exp_approx(vec3 v)
{
    return quat_normalize(quat(1.0f, v.x, v.y, v.z));
}
```

And...

```c++
vec3 quat_log_approx(quat q)
{
    return vec3(q.x, q.y, q.z);
}
```

In fact, these can be quite accurate if we are dealing exclusively with the very small angles/rotations like you might get when integrating angular velocities with a very small `dt`.   

With regards to the clamping before giving as input to `acosf` - that is one solution - but there is actually another way to compute `quat_log` using `atan2` instead:

```c++
vec3 quat_log_alt(quat q, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);
	
    if (length < eps)
    {
        return vec3(q.x, q.y, q.z);
    }
    else
    {
        float halfangle = atan2f(length, q.w);
        return halfangle * (vec3(q.x, q.y, q.z) / length);
    }
}
```

This avoids having to clamp the input between `-1` and `1`, and can be useful in other ways as we will see later on...

Once we have the `log` and `exp` functions implemented it's easy to implement functions for converting to scaled-angle-axis and back.

```c++
quat quat_from_scaled_angle_axis(vec3 v, float eps=1e-8f)
{
    return quat_exp(v / 2.0f, eps);
}
```

And...

```c++
vec3 quat_to_scaled_angle_axis(quat q, float eps=1e-8f)
{
    return 2.0f * quat_log(q, eps);
}
```

Which we can then use for computing and integrating angular velocities:

```c++
vec3 quat_differentiate_angular_velocity(
    quat next, quat curr, float dt, float eps=1e-8f)
{
    return quat_to_scaled_angle_axis(
        quat_abs(quat_mul(next, quat_inv(curr))), eps) / dt; 
}
```

And...

```c++
quat quat_integrate_angular_velocity(
    vec3 vel, quat curr, float dt, float eps=1e-8f)
{
    return quat_mul(quat_from_scaled_angle_axis(vel * dt, eps), curr);
}
```

Where the `quat_abs` function ensure the quaternion is on the hemisphere closest to the identity quaternion, and avoids creating angular velocities that "wrap around" (see [this article](https://www.daniel-holden.com/page/visualizing-rotation-spaces) for an intuitive explanation of why this is required).

```c++
quat quat_abs(quat x)
{
    return x.w < 0.0 ? -x : x;
}
```

Implementations of angle-axis conversion are similar, but we can actually simplify the code a bit for these special cases where the axis is already normalized:

```c++
quat quat_from_angle_axis(float angle, vec3 axis)
{
    float c = cosf(angle / 2.0f);
    float s = sinf(angle / 2.0f);
    return quat(c, s * axis.x, s * axis.y, s * axis.z);
}
```

And...

```c++
void quat_to_angle_axis(quat q, float& angle, vec3& axis, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);

    if (length < eps)
    {
        angle = 0.0f;
        axis = vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        angle = 2.0f * acosf(clampf(q.w, -1.0f, 1.0f));
        axis = vec3(q.x, q.y, q.z) / length;
    }
}
```

Or

```c++
void quat_to_angle_axis_alt(quat q, float& angle, vec3& axis, float eps=1e-8f)
{
    float length = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z);

    if (length < eps)
    {
        angle = 0.0f;
        axis = vec3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        angle = 2.0f * atan2f(length, q.w);	
        axis = vec3(q.x, q.y, q.z) / length;
    }
}
```

From these I think we can also get some more intuition as to what the different numbers in a quaternion actually represent: we can see that they're not so different from the angle axis representation - the `w` component is simply the `cos` of the half angle, while the other components are proportional to the axis, but scaled by the `sin` of the half-angle.

I like to think of this `sin` scaling as just being the normalization which is putting the quaternion onto the unit hypersphere. Ah! Now we can also see how `quat_to_angle_axis` is derived too (at least the `acos` version) - it's using `acos` to get back the half angle, and then re-normalizing the other three components of the quat to get the axis back again. Pretty straight forward!

What about that version using `atan2`? Well in the world of deep learning this version is particularly useful. This is because we might want to back-propagate through these functions and here the `atan2` implementation has the advantage of better gradients than `acos` for small rotations (as the gradients of acos go to `inf` and `-inf` when `w` is either close to `1` or `-1`).

We can also compare the gradients of computing angle-axis vs the exponential map. Here the angle-axis functions completely kill the gradient when it is close to the identity rotation - they simply output some constants - while our `exp` and `log` functions preserve it via their small angle approximation. This makes them better candidates for use in any kind of deep learning.

These gradients however can still be extreme for small angles, since most functions are dividing by the half-angle or length. If we want really clean gradients for small angles we might even want to use only the small angle approximations we derived earlier. e.g. for integrating or differentiating small velocities:

```c++
quat quat_from_scaled_angle_axis_approx(vec3 v)
{
    return quat_exp_approx(v / 2.0f);
}

vec3 quat_to_scaled_angle_axis_approx(quat q)
{
    return 2.0f * quat_log_approx(q);
}

vec3 quat_differentiate_angular_velocity_approx(quat next, quat curr, float dt)
{
    return quat_to_scaled_angle_axis_approx(
        quat_abs(quat_mul(next, quat_inv(curr)))) / dt; 
}

quat quat_integrate_angular_velocity_approx(vec3 vel, quat curr, float dt)
{
    return quat_mul(quat_from_scaled_angle_axis_approx(vel * dt), curr);
}
```

Since these don't have any division their derivatives are very smooth. Useful for performing a differentiable physics integration step perhaps!

And that's all I've got for you today! It turns out the details are important when it comes to the exponential map / angle-axis / angular velocity. I hope this article has got you thinking a bit more deeply about these functions, how they can be implemented, their numerical stability, as well as derivatives.

If you're interesting in more on low level rotation stuff, check out [this post](https://www.daniel-holden.com/page/visualizing-rotation-spaces) on visualizing rotation representations.

---------------------------------------
> 本文出自CaterpillarStudyGroup，转载请注明出处。
>
> https://caterpillarstudygroup.github.io/ImportantArticles/