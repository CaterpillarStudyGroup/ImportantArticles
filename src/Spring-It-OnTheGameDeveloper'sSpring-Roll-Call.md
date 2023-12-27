转载出处：https://www.daniel-holden.com/page/perfect-tracking-springs

 
# Spring-It-On: The Game Developer's Spring-Roll-Call   


## Created on March 4, 2021, 5:49 p.m.   

Springs! What do springs have to do with game development? We'll if you're asking that question and reading this article you're in the right place. Because we're about to do a lot of talking about springs... and, while some of you may well have used springs before, I'm guessing that even if you did the code you used resided in the dark depths of your project as a set of mysterious equations that no one ever touched.

And that's sad, because although the maths can undeniably be tricky, springs are interesting, and a surprisingly versatile tool, with lots of applications in Computer Science that I never even realized were possible until I thought "wouldn't it be nice to know how those equations came about?" and dug a bit deeper.

So I think every Computer Scientist, and in particular those interested in game development, animation, or physics, could probably benefit from a bit of knowledge of springs. In the very least this: what they are, what they do, how they do it, and what they can be used for. So with that in mind, let's start right from the beginning: *The Damper*.

---

## The Damper  

As a game developer here is a situation you've probably come across: for some reason there is an object which suddenly we realize should be at a different position - perhaps we got some information from a game server about where it should really be, or the player did something and now the object needs to be moved. Well, when we move this object we probably don't want it to pop - instead we would prefer if it moved more smoothly towards the new position.

A simple thing we might try to achieve this is to just blend the object's position `x` with the new goal position `g` by some fixed `factor` such as `0.1` each frame.

In C++ we might write something like this, and while this code is for a single `float` the same can be done for a position just by applying it to each component of the vector independently.

```c++
float lerp(float x, float y, float a)
{
    return (1.0f - a) * x + a * y;
}

float damper(float x, float g, float factor)
{
    return lerp(x, g, factor);
}
```

By applying this `damper` function each frame we can smoothly move toward the goal without popping. We can even control the speed of this movement using the `factor` argument:



\begin{align*} x = \text{damper}(x, g,\text{factor}); \end{align*}



Below you can see a visualization of this in action where the horizontal axis represents time and the vertical axis represents the position of the object.

https://www.daniel-holden.com/media/uploads/springs/damper.m4v

But this solution has a problem: if we change the framerate of our game (or the timestep of our system) we get different behavior from the `damper`. More specifically, it moves the object slower when we have a lower framerate:

https://www.daniel-holden.com/media/uploads/springs/damper_dt.m4v


And this makes sense if we think about it - if the game is running at 30 frames per second you are going to perform half as many calls to `damper` as if you were running at 60 frames per second, so the object is not going to get pulled toward the goal as quickly. One simple idea for a fix might be to just multiply the factor by the timestep `dt` - now at least when the timestep is larger the object will move more quickly toward the goal...


```c++
float damper_bad(float x, float t, float damping, float dt)
{
    return lerp(x, t, damping * dt);
}
```

This might appear like it works on face value but there are two big problems with this solution which can come back to bite us badly. Firstly, we now have a mysterious `damping` variable which is difficult to set and interpret. But secondly, and more importantly, if we set the `damping` or the `dt` too high (such that `damping * dt > 1`) the whole thing becomes unstable, and in the worst case explodes:

https://www.daniel-holden.com/media/uploads/springs/damper_bad.m4v

We could use various hacks like clamping `damping * dt` to be less than `1` but there is fundamentally something wrong with what we've done here. We can see this if we imagine that `damping * dt` is roughly equal to `0.5` - here, doubling the `dt` does not produce the same result as applying the damper twice: lerping with a factor of `0.5` twice will take us 75% of the way toward the goal, while lerping with a `factor` of `1.0` once will bring us 100% of the way there. So what's the real fix?

--- 

## The Exact Damper  

Let's start our investigation by plotting the behavior of `x` using the normal `damper` with a fixed `dt` of `1.0`, a `goal` of `0`, and a `factor` of `0.5`:

![](./assets/06-1.png)


Here we can see repeated calls to `lerp` actually produce a kind of exponential decay toward the goal:

\begin{align*} t=0,x=1.0 \\\\ t=1, & x=0.5 \\\\t=2, & x=0.25 \\\\t=3, & x=0.125 \end{align*}

 
And for a `lerp` factor of `0.5`, we can see that this pattern is exactly the equation `\\(x_t=0.5^t\\)` . So it looks like somehow there is an exponential function governing this relationship, but how did this appear? The trick to uncovering this exponential form is to write our system as a recurrence relation.

## Recurrence Relation   

We'll start by defining a separate variable `\\(y=1−damping⋅ft\\)`, which will make the maths a bit easier later on. In this case `\\(ft\\)` is a fixed, small `\\(dt\\)` such as `\\( \frac{1}{60} \\)`. Then we will expand the `lerp` function:

\begin{align*} t=0, & x=1.0 \\\\ t=1, & x=0.5 \\\\t=2, & x=0.25 \\\\t=3, & x=0.125 \end{align*}

  
=lerp(x 
t
​
 ,g,1−y)
=(1−(1−y))⋅x 
t
​
 +(1−y)⋅g
=y⋅x 
t
​
 −(y−1)⋅g
=y⋅x 
t
​
 −y⋅g+g
​
 

Now for the recurrence relation: by plugging this equation into itself we are going to see how the exponent appears. First we need to increment 
�
+
1
t+1 to 
�
+
2
t+2 and then replace the new 
�
�
+
1
x 
t+1
​
  which appears on the right hand side with the same equation again.

�
�
+
1
=
�
⋅
�
�
−
�
⋅
�
+
�
�
�
+
2
=
�
⋅
�
�
+
1
−
�
⋅
�
+
�
�
�
+
2
=
�
⋅
(
�
⋅
�
�
−
�
⋅
�
+
�
)
−
�
⋅
�
+
�
�
�
+
2
=
�
⋅
�
⋅
�
�
−
�
⋅
�
⋅
�
+
�
⋅
�
−
�
⋅
�
+
�
�
�
+
2
=
�
⋅
�
⋅
�
�
−
�
⋅
�
⋅
�
+
�
x 
t+1
​
 
x 
t+2
​
 
x 
t+2
​
 
x 
t+2
​
 
x 
t+2
​
 
​
  
=y⋅x 
t
​
 −y⋅g+g
=y⋅x 
t+1
​
 −y⋅g+g
=y⋅(y⋅x 
t
​
 −y⋅g+g)−y⋅g+g
=y⋅y⋅x 
t
​
 −y⋅y⋅g+y⋅g−y⋅g+g
=y⋅y⋅x 
t
​
 −y⋅y⋅g+g
​
 

If we repeat this process again and we can start to see a pattern emerging:

�
�
+
2
=
�
⋅
�
⋅
�
�
−
�
⋅
�
⋅
�
+
�
�
�
+
3
=
�
⋅
�
⋅
�
�
+
1
−
�
⋅
�
⋅
�
+
�
�
�
+
3
=
�
⋅
�
⋅
(
�
⋅
�
�
−
�
⋅
�
+
�
)
−
�
⋅
�
⋅
�
+
�
�
�
+
3
=
�
⋅
�
⋅
�
⋅
�
�
−
�
⋅
�
⋅
�
⋅
�
+
�
⋅
�
⋅
�
−
�
⋅
�
⋅
�
+
�
�
�
+
3
=
�
⋅
�
⋅
�
⋅
�
�
−
�
⋅
�
⋅
�
⋅
�
+
�
x 
t+2
​
 
x 
t+3
​
 
x 
t+3
​
 
x 
t+3
​
 
x 
t+3
​
 
​
  
=y⋅y⋅x 
t
​
 −y⋅y⋅g+g
=y⋅y⋅x 
t+1
​
 −y⋅y⋅g+g
=y⋅y⋅(y⋅x 
t
​
 −y⋅g+g)−y⋅y⋅g+g
=y⋅y⋅y⋅x 
t
​
 −y⋅y⋅y⋅g+y⋅y⋅g−y⋅y⋅g+g
=y⋅y⋅y⋅x 
t
​
 −y⋅y⋅y⋅g+g
​
 

More generally we can see that:

�
�
+
�
=
�
�
⋅
�
�
−
�
�
⋅
�
+
�
x 
t+n
​
 =y 
n
 ⋅x 
t
​
 −y 
n
 ⋅g+g
​
 

Ah-ha! Our exponent has appeared. And by rearranging a bit we can even write this in terms of 
lerp
lerp again:

�
�
+
�
=
�
�
⋅
�
�
−
�
�
⋅
�
+
�
�
�
+
�
=
�
�
⋅
�
�
+
�
⋅
(
1
−
�
�
)
�
�
+
�
=
lerp
(
�
�
,
�
,
1
−
�
�
)
x 
t+n
​
 
x 
t+n
​
 
x 
t+n
​
 
​
  
=y 
n
 ⋅x 
t
​
 −y 
n
 ⋅g+g
=y 
n
 ⋅x 
t
​
 +g⋅(1−y 
n
 )
=lerp(x 
t
​
 ,g,1−y 
n
 )
​
 

As a small tweak, we can make the exponent negative:

�
�
+
�
=
lerp
(
�
�
,
�
,
1
−
�
�
)
�
�
+
�
=
lerp
(
�
�
,
�
,
1
−
1
�
−
�
)
x 
t+n
​
 
x 
t+n
​
 
​
  
=lerp(x 
t
​
 ,g,1−y 
n
 )
=lerp(x 
t
​
 ,g,1− 
y
1
​
  
−n
 )
​
 

Remember that 
�
n represents a multiple of 
�
�
ft, so if we have a new arbitrary 
�
�
dt we will need to convert it to 
�
n first using 
�
=
�
�
�
�
n= 
ft
dt
​
 . In C++ we would write it as follows:

float damper_exponential(
    float x,
    float g, 
    float damping, 
    float dt, 
    float ft = 1.0f / 60.0f)
{
    return lerp(x, g, 1.0f - powf(1.0 / (1.0 - ft * damping), -dt / ft));
} 
Let's see it action! Notice how it produces the same, identical and stable behavior even when we make the dt and damping large.


So have we fixed it? Well, in this formulation we've essentially solved the problem by letting the behavior of the damper match one particular timestep while allowing the rate of decay to still vary. In this case 1.0f - ft * damping is our rate of decay, and it dictates what proportion of the distance toward the goal will remain after ft in time. As long as we make the fixed timestep ft small enough, ft * damping should never exceed 1.0 and the system remains stable and well behaved.

The Half-Life
But there is another, potentially better way to fix the problem. Instead of fixing the timestep, we can fix the rate of decay and let the timestep vary. This sounds a little odd at first but in practice it makes things much easier. The basic idea is simple: let's set the rate of decay to 
0.5
0.5 and instead scale the timestep such that we can control the exact half-life of the damper - a.k.a the time it takes for the distance to the goal to reduce by half:

�
�
+
�
�
=
lerp
(
�
�
,
�
,
1
−
1
0.5
−
�
�
/
ℎ
�
�
�
�
�
�
�
)
�
�
+
�
�
=
lerp
(
�
�
,
�
,
1
−
2
−
�
�
/
ℎ
�
�
�
�
�
�
�
)
x 
t+dt
​
 
x 
t+dt
​
 
​
  
=lerp(x 
t
​
 ,g,1− 
0.5
1
​
  
−dt/halflife
 )
=lerp(x 
t
​
 ,g,1−2 
−dt/halflife
 )
​
 

This simplifies the code and gives a more intuitive parameter to control the damper. Now we don't ever need to worry about if we've set the damping too large or made the fixed timestep ft small enough.

float damper_exact(float x, float g, float halflife, float dt)
{
    return lerp(x, g, 1.0f - powf(2, -dt / halflife));
}
For neatness, we can also switch to an exponential base using the change of base theorem: just multiply the dt by 
�
�
(
2
)
=
0.69314718056
ln(2)=0.69314718056 and switch to using expf. Finally, we should add some small epsilon value like 1e-5f to avoid division by zero when our halflife is very small:

float damper_exact(float x, float g, float halflife, float dt, float eps=1e-5f)
{
    return lerp(x, g, 1.0f - expf(-(0.69314718056f * dt) / (halflife + eps)));
}
The change of base theorem tells us another thing: that changing the rate of decay is no different from scaling the dt in the exponent. So using the halflife to control the damper should not limit us in any of the behaviors we want to achieve compared to if we changed the rate of decay like in our previous setup.

There is one more nice little trick we can do - a fast approximation of the negative exponent function using one over a simple polynomial (or we could use this even better approximation from Danny Chapman):

float fast_negexp(float x)
{
    return 1.0f / (1.0f + x + 0.48f*x*x + 0.235f*x*x*x);
}
And that's it - we've converted our unstable damper into one that is fast, stable, and has intuitive parameters!

float damper_exact(float x, float g, float halflife, float dt, float eps=1e-5f)
{
    return lerp(x, g, 1.0f-fast_negexp((0.69314718056f * dt) / (halflife + eps)));
}
Let's see how it looks...


Perfect!

The Spring Damper
The exact damper works well in a lot of cases, but has one major issue - it creates discontinuities when the goal position changes quickly. For example, even if the object is moving in one direction, it will immediately switch to moving in the opposite direction if the goal changes direction. This can create a kind of annoying sudden movement which you can see in the previous videos.

The problem is that there is no velocity continuity - no matter what happened in the previous frames the damper will always move toward the goal. Let's see how we might be able to fix that. We can start by looking again at our old broken bad damper, and examining it in a bit more detail:

�
�
+
�
�
=
lerp
(
�
�
,
�
,
�
�
⋅
�
�
�
�
�
�
�
)
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
x 
t+dt
​
 
x 
t+dt
​
 
​
  
=lerp(x 
t
​
 ,g,dt⋅damping)
=x 
t
​
 +dt⋅damping⋅(g−x 
t
​
 )
​
 

We can see that this looks a bit like a physics equation where 
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
damping⋅(g−x 
t
​
 ) represents the velocity.

�
�
=
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
v 
t+dt
​
 
x 
t+dt
​
 
​
  
=damping⋅(g−x 
t
​
 )
=x 
t
​
 +dt⋅v 
t
​
 
​
 

This system is like a kind of particle with a velocity always proportional to the difference between the current particle position and the goal position. This explains the discontinuity - the velocity of our damper will always be directly proportional to the difference between the current position and the goal without ever taking any previous velocities into account.

What if instead of setting the velocity directly each step we made it something that changed more smoothly? For example, we could instead add a velocity taking us toward the goal to the current velocity, scaled by a different parameter which for now we will call the 
�
�
�
�
�
�
�
�
�
stiffness.

�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
v 
t+dt
​
 
x 
t+dt
​
 
​
  
=v 
t
​
 +dt⋅stiffness⋅(g−x 
t
​
 )
=x 
t
​
 +dt⋅v 
t
​
 
​
 

But the problem now is that this particle wont slow down until it has already over-shot the goal and is pulled back in the opposite direction. To fix this we can add a 
�
q variable which represents a goal velocity, and add another term which takes us toward this goal velocity. This we will scale by another new parameter which we will call the 
�
�
�
�
�
�
�
damping (for reasons which will become clearer later in the article).

�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
+
�
�
⋅
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
v 
t+dt
​
 
x 
t+dt
​
 
​
  
=v 
t
​
 +dt⋅stiffness⋅(g−x 
t
​
 )+dt⋅damping⋅(q−v 
t
​
 )
=x 
t
​
 +dt⋅v 
t
​
 
​
 

When 
�
q is very small we can think of this like a kind of friction term which simply subtracts the current velocity. And when 
�
=
0
q=0 and 
�
�
⋅
�
�
�
�
�
�
�
=
1
dt⋅damping=1 we can see that this friction term actually completely removes the existing velocity, reverting our system back to something just like our original damper.

Another way to think about these terms is by thinking of them as accelerations, which can be shown more clearly by factoring out the 
�
�
dt:

�
�
=
�
�
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
+
�
�
�
�
�
�
�
⋅
(
�
−
�
�
)
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
�
�
+
�
�
=
�
�
+
�
�
⋅
�
�
a 
t+dt
​
 
v 
t+dt
​
 
x 
t+dt
​
 
​
  
=stiffness⋅(g−x 
t
​
 )+damping⋅(q−v 
t
​
 )
=v 
t
​
 +dt⋅a 
t
​
 
=x 
t
​
 +dt⋅v 
t
​
 
​
 

Assuming the mass of our particle is exactly one, it really is possible to think about this as two individual forces - one pulling the particle in the direction of the goal velocity, and one pulling it toward the goal position. If we use a small enough 
�
�
dt we can actually plug these functions together and simulate a simple damped spring with exactly the velocity continuity we wanted. Here is a function which does that (using semi-implicit euler integration).

void spring_damper_bad(
    float& x,
    float& v, 
    float g,
    float q, 
    float stiffness, 
    float damping, 
    float dt)
{
    v += dt * stiffness * (g - x) + dt * damping * (q - v);
    x += dt * v;
}
Let's see how it looks:


But unfortunately just like before we have problems when the 
�
�
dt is large, and certain settings for 
�
�
�
�
�
�
�
�
�
stiffness and 
�
�
�
�
�
�
�
damping can make the system unstable. These unintuitive parameters like 
�
�
�
�
�
�
�
damping and 
�
�
�
�
�
�
�
�
�
stiffness are also back again... arg!


Can we give this spring the same treatment as we did for our damper by fiddling around with the maths? Well yes we can, but unfortunately from here on in things are going to get a bit more complicated...

The Exact Spring Damper
This time the exact version of our model is too complicated to solve using a simple recurrence relation. Instead we're going to have to try a different tactic: we're going to guess an equation we think models the spring and then try to work out how to compute all the different parameters of that equation based on the parameters we do know such as the 
�
�
�
�
�
�
�
damping and 
�
�
�
�
�
�
�
�
�
stiffness.

If we take a look at the movement of the spring in the previous section we can see there are basically two features - an exponential decay toward the goal position, and a kind of oscillation, a bit like a 
cos
⁡
cos or 
sin
⁡
sin function. So let's try and come up with an equation which fits that kind of shape and go from there. What about something like this?

�
�
=
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
+
�
x 
t
​
 =j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)+c
​
 

Where 
�
j is the amplitude, 
�
y controls the time it takes to decay, a bit like our half-life parameter, 
�
t is the time, 
�
w is the frequency of oscillations, 
�
p is the phase of oscillations, and 
�
c is an offset on the vertical axis. This seems like a reasonable formulation of the behavior we saw previously.

But before we try to find all of these unknown parameters, let's write down the derivatives of this function with respect to 
�
t too. We'll use 
�
�
v 
t
​
  to denote the velocity, and 
�
�
a 
t
​
  to denote the acceleration.

�
�
=
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
+
�
�
�
=
−
�
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
−
�
⋅
�
⋅
�
−
�
⋅
�
⋅
sin
⁡
(
�
⋅
�
+
�
)
�
�
=
�
2
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
−
�
2
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
+
2
⋅
�
⋅
�
⋅
�
⋅
�
−
�
⋅
�
⋅
sin
⁡
(
�
⋅
�
+
�
)
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)+c
=−y⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
−w⋅j⋅e 
−y⋅t
 ⋅sin(w⋅t+p)
=y 
2
 ⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
−w 
2
 ⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
+2⋅w⋅y⋅j⋅e 
−y⋅t
 ⋅sin(w⋅t+p)
​
 

Those might look a bit scary but we can make them a lot less scary by just summarizing some of the common terms:

�
=
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
�
=
�
⋅
�
−
�
⋅
�
⋅
sin
⁡
(
�
⋅
�
+
�
)
C
S
​
  
=j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
=j⋅e 
−y⋅t
 ⋅sin(w⋅t+p)
​
 

Giving us the following:

�
�
=
�
+
�
�
�
=
−
�
⋅
�
−
�
⋅
�
�
�
=
�
2
⋅
�
−
�
2
⋅
�
+
2
⋅
�
⋅
�
⋅
�
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=C+c
=−y⋅C−w⋅S
=y 
2
 ⋅C−w 
2
 ⋅C+2⋅w⋅y⋅S
​
 

Finding the Spring Parameters
Our plan for finding the first set of unknown parameters is as follows: we're going to substitute these new equations for 
�
�
x 
t
​
 , 
�
�
v 
t
​
 , and 
�
�
a 
t
​
  into our previous equation of motion 
�
�
=
�
⋅
(
�
−
�
�
)
+
�
⋅
(
�
−
�
�
)
a 
t
​
 =s⋅(g−x 
t
​
 )+d⋅(q−v 
t
​
 ) (where 
�
=
�
�
�
�
�
�
�
d=damping and 
�
=
�
�
�
�
�
�
�
�
�
s=stiffness ) and try to rearrange to solve for 
�
y, 
�
w, and 
�
c using all the other values we know: 
�
s, 
�
d, 
�
q, and 
�
g.

But first let's shuffle around some terms in this equation of motion: expanding the 
�
�
�
�
�
�
�
�
�
stiffness and 
�
�
�
�
�
�
�
damping terms, moving some values onto the left hand side, and finally negating everything. This will make the next steps much easier for us.

�
�
=
�
⋅
(
�
−
�
�
)
+
�
⋅
(
�
−
�
�
)
0
=
�
⋅
(
�
−
�
�
)
+
�
⋅
(
�
−
�
�
)
−
�
�
0
=
�
⋅
�
−
�
⋅
�
�
+
�
⋅
�
−
�
⋅
�
�
−
�
�
−
�
⋅
�
−
�
⋅
�
=
−
�
⋅
�
�
−
�
⋅
�
�
−
�
�
�
⋅
�
+
�
⋅
�
=
�
⋅
�
�
+
�
⋅
�
�
+
�
�
a 
t
​
 
0
0
−s⋅g−d⋅q
s⋅g+d⋅q
​
  
=s⋅(g−x 
t
​
 )+d⋅(q−v 
t
​
 )
=s⋅(g−x 
t
​
 )+d⋅(q−v 
t
​
 )−a 
t
​
 
=s⋅g−s⋅x 
t
​
 +d⋅q−d⋅v 
t
​
 −a 
t
​
 
=−s⋅x 
t
​
 −d⋅v 
t
​
 −a 
t
​
 
=s⋅x 
t
​
 +d⋅v 
t
​
 +a 
t
​
 
​
 

Now let's substitute in our three new equations we just created for 
�
�
x 
t
​
 , 
�
�
v 
t
​
 , and 
�
�
a 
t
​
 :

�
⋅
�
+
�
⋅
�
=
�
⋅
�
�
+
�
⋅
�
�
+
�
�
�
⋅
�
+
�
⋅
�
=
�
⋅
(
�
+
�
)
+
�
⋅
(
−
�
⋅
�
−
�
⋅
�
)
+
(
(
�
2
−
�
2
)
⋅
�
+
2
⋅
�
⋅
�
⋅
�
)
s⋅g+d⋅q
s⋅g+d⋅q
​
  
=s⋅x 
t
​
 +d⋅v 
t
​
 +a 
t
​
 
=s⋅(C+c)+d⋅(−y⋅C−w⋅S)+((y 
2
 −w 
2
 )⋅C+2⋅w⋅y⋅S)
​
 

And by multiplying out and then gathering all the coefficients of 
�
C and 
�
S together we can get:

�
⋅
�
+
�
⋅
�
=
�
⋅
(
�
+
�
)
+
�
⋅
(
−
�
⋅
�
−
�
⋅
�
)
+
(
(
�
2
−
�
2
)
⋅
�
+
2
⋅
�
⋅
�
⋅
�
)
�
⋅
�
+
�
⋅
�
−
�
⋅
�
=
�
⋅
�
+
�
⋅
−
�
⋅
�
−
�
⋅
�
⋅
�
+
�
2
⋅
�
−
�
2
⋅
�
+
2
⋅
�
⋅
�
⋅
�
�
⋅
�
+
�
⋅
�
−
�
⋅
�
=
(
(
�
2
−
�
2
)
−
�
⋅
�
+
�
)
⋅
�
+
(
2
⋅
�
⋅
�
−
�
⋅
�
)
⋅
�
s⋅g+d⋅q
s⋅g+d⋅q−s⋅c
s⋅g+d⋅q−s⋅c
​
  
=s⋅(C+c)+d⋅(−y⋅C−w⋅S)+((y 
2
 −w 
2
 )⋅C+2⋅w⋅y⋅S)
=s⋅C+d⋅−y⋅C−d⋅w⋅S+y 
2
 ⋅C−w 
2
 ⋅C+2⋅w⋅y⋅S
=((y 
2
 −w 
2
 )−d⋅y+s)⋅C+(2⋅w⋅y−d⋅w)⋅S
​
 

There is one more additional fact we can use to get the variables we need from this equation: because 
�
C and 
�
S are essentially 
cos
⁡
cos and 
sin
⁡
sin functions with the same phase, amplitude, and frequency, the only way this equation can be balanced for all potential values of 
�
t, 
�
w, 
�
y, 
�
j and 
�
c is when both the coefficients of 
�
C and 
�
S are zero and when the left hand side equals zero. This gives us three smaller equations to solve:

�
⋅
�
+
�
⋅
�
−
�
⋅
�
=
0
(
�
2
−
�
2
)
−
�
⋅
�
+
�
=
0
2
⋅
�
⋅
�
−
�
⋅
�
=
0
s⋅g+d⋅q−s⋅c
(y 
2
 −w 
2
 )−d⋅y+s
2⋅w⋅y−d⋅w
​
  
=0
=0
=0
​
  
​
 

Finding 
�
c
Using equation 
(
1
)
(1) we can solve for 
�
c right away to get our first unknown!

�
⋅
�
+
�
⋅
�
−
�
⋅
�
=
0
�
⋅
�
+
�
⋅
�
=
�
⋅
�
�
+
�
⋅
�
�
=
�
s⋅g+d⋅q−s⋅c
s⋅g+d⋅q
g+ 
s
d⋅q
​
 
​
  
=0
=s⋅c
=c
​
 

Finding 
�
y
And by rearranging equation 
(
3
)
(3) we can also find a solution for 
�
y:

2
⋅
�
⋅
�
−
�
⋅
�
=
0
�
⋅
�
=
2
⋅
�
⋅
�
�
=
2
⋅
�
�
2
=
�
2⋅w⋅y−d⋅w
d⋅w
d
2
d
​
 
​
  
=0
=2⋅w⋅y
=2⋅y
=y
​
 

Finding 
�
w
Which we can substitute into equation 
(
2
)
(2) to solve for 
�
w:

(
�
2
−
�
2
)
−
�
⋅
�
+
�
=
0
(
(
�
2
)
2
−
�
2
)
−
�
⋅
�
2
+
�
=
0
�
2
4
−
�
2
−
�
2
2
+
�
=
0
�
2
4
−
�
2
2
+
�
=
�
2
�
−
�
2
4
=
�
2
�
−
�
2
4
=
�
(y 
2
 −w 
2
 )−d⋅y+s
(( 
2
d
​
 ) 
2
 −w 
2
 )−d⋅ 
2
d
​
 +s
4
d 
2
 
​
 −w 
2
 − 
2
d 
2
 
​
 +s
4
d 
2
 
​
 − 
2
d 
2
 
​
 +s
s− 
4
d 
2
 
​
 
s− 
4
d 
2
 
​
 
​
 
​
  
=0
=0
=0
=w 
2
 
=w 
2
 
=w
​
 

Finding the Spring State
There are two final unknown variables remaining: 
�
j, and 
�
p - the amplitude and the phase. Unlike 
�
y, 
�
w, and 
�
c, these two are determined by the initial conditions of the spring. Therefore, given some initial position and velocity, 
�
0
x 
0
​
  and 
�
0
v 
0
​
 , we can plug these in our equations along with 
�
=
0
t=0 to get some more equations we will use to find 
�
j and 
�
p:

�
0
=
�
⋅
�
−
�
⋅
0
⋅
cos
⁡
(
�
⋅
0
+
�
)
+
�
�
0
=
�
⋅
cos
⁡
(
�
)
+
�
�
0
=
−
�
⋅
�
⋅
�
−
�
⋅
0
⋅
cos
⁡
(
�
⋅
0
+
�
)
−
�
⋅
�
⋅
�
−
�
⋅
0
⋅
sin
⁡
(
�
⋅
0
+
�
)
�
0
=
−
�
⋅
�
⋅
cos
⁡
(
�
)
−
�
⋅
�
⋅
sin
⁡
(
�
)
x 
0
​
 
x 
0
​
 
v 
0
​
 
v 
0
​
 
​
  
=j⋅e 
−y⋅0
 ⋅cos(w⋅0+p)+c
=j⋅cos(p)+c
=−y⋅j⋅e 
−y⋅0
 ⋅cos(w⋅0+p)−w⋅j⋅e 
−y⋅0
 ⋅sin(w⋅0+p)
=−y⋅j⋅cos(p)−w⋅j⋅sin(p)
​
 

Finding 
�
j
Let's start with 
�
j. First we'll re-arrange our equation for 
�
0
x 
0
​
  in terms of 
�
p:

�
0
=
�
⋅
cos
⁡
(
�
)
+
�
�
0
−
�
=
�
⋅
cos
⁡
(
�
)
�
0
−
�
�
=
cos
⁡
(
�
)
arccos
⁡
(
�
0
−
�
�
)
=
�
x 
0
​
 
x 
0
​
 −c
j
x 
0
​
 −c
​
 
arccos( 
j
x 
0
​
 −c
​
 )
​
  
=j⋅cos(p)+c
=j⋅cos(p)
=cos(p)
=p
​
 

And substitute this into our equation for 
�
0
v 
0
​
 :

�
0
=
−
�
⋅
�
⋅
cos
⁡
(
�
)
−
�
⋅
�
⋅
sin
⁡
(
�
)
�
0
=
−
�
⋅
�
⋅
cos
⁡
(
arccos
⁡
(
�
0
−
�
�
)
)
−
�
⋅
�
⋅
sin
⁡
(
arccos
⁡
(
�
0
−
�
�
)
)
�
0
=
−
�
⋅
�
⋅
�
0
−
�
�
−
�
⋅
�
⋅
1
−
(
�
0
−
�
)
2
�
2
�
0
=
−
�
⋅
(
�
0
−
�
)
−
�
⋅
�
⋅
1
−
(
�
0
−
�
)
2
�
2
v 
0
​
 
v 
0
​
 
v 
0
​
 
v 
0
​
 
​
  
=−y⋅j⋅cos(p)−w⋅j⋅sin(p)
=−y⋅j⋅cos(arccos( 
j
x 
0
​
 −c
​
 ))−w⋅j⋅sin(arccos( 
j
x 
0
​
 −c
​
 ))
=−y⋅j⋅ 
j
x 
0
​
 −c
​
 −w⋅j⋅ 
1− 
j 
2
 
(x 
0
​
 −c) 
2
 
​
 
​
 
=−y⋅(x 
0
​
 −c)−w⋅j⋅ 
1− 
j 
2
 
(x 
0
​
 −c) 
2
 
​
 
​
 
​
 

Which we can now rearrange for 
�
j:

�
0
+
�
⋅
(
�
0
−
�
)
=
−
�
⋅
�
⋅
1
−
(
�
0
−
�
)
2
�
2
�
0
+
�
⋅
(
�
0
−
�
)
−
�
⋅
�
=
1
−
(
�
0
−
�
)
2
�
2
(
�
0
+
�
⋅
(
�
0
−
�
)
)
2
(
−
�
⋅
�
)
2
=
1
−
(
�
0
−
�
)
2
�
2
(
�
0
+
�
⋅
(
�
0
−
�
)
)
2
�
2
=
�
2
−
(
�
0
−
�
)
2
(
�
0
+
�
⋅
(
�
0
−
�
)
)
2
�
2
+
(
�
0
−
�
)
2
=
�
2
(
�
0
+
�
⋅
(
�
0
−
�
)
)
2
�
2
+
(
�
0
−
�
)
2
=
�
v 
0
​
 +y⋅(x 
0
​
 −c)
−w⋅j
v 
0
​
 +y⋅(x 
0
​
 −c)
​
 
(−w⋅j) 
2
 
(v 
0
​
 +y⋅(x 
0
​
 −c)) 
2
 
​
 
w 
2
 
(v 
0
​
 +y⋅(x 
0
​
 −c)) 
2
 
​
 
w 
2
 
(v 
0
​
 +y⋅(x 
0
​
 −c)) 
2
 
​
 +(x 
0
​
 −c) 
2
 
w 
2
 
(v 
0
​
 +y⋅(x 
0
​
 −c)) 
2
 
​
 +(x 
0
​
 −c) 
2
 
​
 
​
  
=−w⋅j⋅ 
1− 
j 
2
 
(x 
0
​
 −c) 
2
 
​
 
​
 
= 
1− 
j 
2
 
(x 
0
​
 −c) 
2
 
​
 
​
 
=1− 
j 
2
 
(x 
0
​
 −c) 
2
 
​
 
=j 
2
 −(x 
0
​
 −c) 
2
 
=j 
2
 
=j
​
 

Nice! Since this relies on squares and a square root, some sign information is lost. This means that in our code we will also need to negate 
�
j in the case that 
�
0
−
�
<
0
x 
0
​
 −c<0.

Finding 
�
p
Finally, we are ready to find 
�
p. We can start by rearranging our velocity equation 
�
0
v 
0
​
  for 
�
j:

�
0
=
−
�
⋅
�
⋅
cos
⁡
(
�
)
−
�
⋅
�
⋅
sin
⁡
(
�
)
�
0
=
�
⋅
(
−
�
⋅
cos
⁡
(
�
)
−
�
⋅
sin
⁡
(
�
)
)
�
0
−
�
⋅
cos
⁡
(
�
)
−
�
⋅
sin
⁡
(
�
)
=
�
v 
0
​
 
v 
0
​
 
−y⋅cos(p)−w⋅sin(p)
v 
0
​
 
​
 
​
  
=−y⋅j⋅cos(p)−w⋅j⋅sin(p)
=j⋅(−y⋅cos(p)−w⋅sin(p))
=j
​
 

And then substitute this into our equation for 
�
0
x 
0
​
  to solve for 
�
p:

�
0
=
�
⋅
cos
⁡
(
�
)
+
�
�
0
=
(
�
0
−
�
⋅
cos
⁡
(
�
)
−
�
⋅
sin
⁡
(
�
)
)
⋅
cos
⁡
(
�
)
+
�
�
0
−
�
=
�
0
⋅
cos
⁡
(
�
)
−
�
⋅
cos
⁡
(
�
)
−
�
⋅
sin
⁡
(
�
)
�
0
−
�
=
�
0
−
�
−
�
⋅
sin
⁡
(
�
)
cos
⁡
(
�
)
�
0
−
�
=
�
0
−
�
−
�
⋅
tan
⁡
(
�
)
(
�
0
−
�
)
⋅
(
−
�
−
�
⋅
tan
⁡
(
�
)
)
=
�
0
−
(
�
0
−
�
)
⋅
�
−
(
�
0
−
�
)
⋅
�
⋅
tan
⁡
(
�
)
=
�
0
−
(
�
0
−
�
)
⋅
�
⋅
tan
⁡
(
�
)
=
�
0
+
(
�
0
−
�
)
⋅
�
tan
⁡
(
�
)
=
�
0
+
(
�
0
−
�
)
⋅
�
−
(
�
0
−
�
)
⋅
�
�
=
arctan
⁡
(
�
0
+
(
�
0
−
�
)
⋅
�
−
(
�
0
−
�
)
⋅
�
)
x 
0
​
 
x 
0
​
 
x 
0
​
 −c
x 
0
​
 −c
x 
0
​
 −c
(x 
0
​
 −c)⋅(−y−w⋅tan(p))
−(x 
0
​
 −c)⋅y−(x 
0
​
 −c)⋅w⋅tan(p)
−(x 
0
​
 −c)⋅w⋅tan(p)
tan(p)
p
​
  
=j⋅cos(p)+c
=( 
−y⋅cos(p)−w⋅sin(p)
v 
0
​
 
​
 )⋅cos(p)+c
= 
−y⋅cos(p)−w⋅sin(p)
v 
0
​
 ⋅cos(p)
​
 
= 
−y−w⋅ 
cos(p)
sin(p)
​
 
v 
0
​
 
​
 
= 
−y−w⋅tan(p)
v 
0
​
 
​
 
=v 
0
​
 
=v 
0
​
 
=v 
0
​
 +(x 
0
​
 −c)⋅y
= 
−(x 
0
​
 −c)⋅w
v 
0
​
 +(x 
0
​
 −c)⋅y
​
 
=arctan( 
−(x 
0
​
 −c)⋅w
v 
0
​
 +(x 
0
​
 −c)⋅y
​
 )
​
 

Putting it together
Putting all of this together, and throwing in a fast approximate atanf for fun, we get the following...

float fast_atan(float x)
{
    float z = fabs(x);
    float w = z > 1.0f ? 1.0f / z : z;
    float y = (M_PI / 4.0f)*w - w*(w - 1)*(0.2447f + 0.0663f*w);
    return copysign(z > 1.0f ? M_PI / 2.0 - y : y, x);
}

float squaref(float x)
{
    return x*x;
}

void spring_damper_exact(
    float& x, 
    float& v, 
    float x_goal, 
    float v_goal, 
    float stiffness, 
    float damping, 
    float dt, 
    float eps = 1e-5f)
{
    float g = x_goal;
    float q = v_goal;
    float s = stiffness;
    float d = damping;
    float c = g + (d*q) / (s + eps);
    float y = d / 2.0f;
    float w = sqrtf(s - (d*d)/4.0f);
    float j = sqrtf(squaref(v + y*(x - c)) / (w*w + eps) + squaref(x - c));
    float p = fast_atan((v + (x - c) * y) / (-(x - c)*w + eps));

    j = (x - c) > 0.0f ? j : -j;

    float eydt = fast_negexp(y*dt);

    x = j*eydt*cosf(w*dt + p) + c;
    v = -y*j*eydt*cosf(w*dt + p) - w*j*eydt*sinf(w*dt + p);
}
Phew - that was a lot of equations and re-arranging, but it worked, and produces a smooth, stable motion even with a very large dt or stiffness. And anyway, doesn't it feel nice to actually use those high school trig identities and do some old school equation manipulation for once!


Over, Under, and Critical Damping
But hold on... one of the steps we took in the previous section wasn't really legit... can you spot it? Here is the problem:

�
=
�
−
�
2
4
w= 
s− 
4
d 
2
 
​
 
​
 
​
 

It's a square root... but I never assured you the input to this square root couldn't be negative. In fact it can be... and definitely will be if 
�
d is large!

But what does this negative square root actually correspond to? Does it mean that there is no exact solution to this spring when the 
�
�
�
�
�
�
�
damping is large? Do we just have to give up? Well, not exactly...

In fact we didn't notice when we came up with our original equation to model the behavior of the spring, but there are three different ways this spring can act depending on the relative sizes of the 
�
�
�
�
�
�
�
damping and 
�
�
�
�
�
�
�
�
�
stiffness values.

If 
�
−
�
2
4
>
0
s− 
4
d 
2
 
​
 >0 it means the spring is under damped, causing oscillations to appear with motions governed by the equations we already derived. If 
�
−
�
2
4
=
0
s− 
4
d 
2
 
​
 =0 it means the spring is critically damped, meaning it returns to the goal as fast as possible without extra oscillation, and if 
�
−
�
2
4
<
0
s− 
4
d 
2
 
​
 <0 it means the spring is over damped, and will return slowly toward the goal.

In each of these cases there is a different set of basic equations governing the system, leading to a different derivation just like the one we completed. I'm going to save us a bit of time and write them all our here rather than going through the trial and error process of examining different guesses at equations and seeing if they fit:

Under Damped:
�
�
=
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
+
�
�
�
=
−
�
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
−
�
⋅
�
⋅
�
−
�
⋅
�
⋅
sin
⁡
(
�
⋅
�
+
�
)
�
�
=
�
2
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
−
�
2
⋅
�
⋅
�
−
�
⋅
�
⋅
cos
⁡
(
�
⋅
�
+
�
)
+
2
⋅
�
⋅
�
⋅
�
⋅
�
−
�
⋅
�
⋅
sin
⁡
(
�
⋅
�
+
�
)
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)+c
=−y⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
−w⋅j⋅e 
−y⋅t
 ⋅sin(w⋅t+p)
=y 
2
 ⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
−w 
2
 ⋅j⋅e 
−y⋅t
 ⋅cos(w⋅t+p)
+2⋅w⋅y⋅j⋅e 
−y⋅t
 ⋅sin(w⋅t+p)
​
 

Critically Damped:
�
�
=
�
0
⋅
�
−
�
⋅
�
+
�
⋅
�
1
⋅
�
−
�
⋅
�
+
�
�
�
=
−
�
⋅
�
0
⋅
�
−
�
⋅
�
−
�
⋅
�
⋅
�
1
⋅
�
−
�
⋅
�
+
�
1
⋅
�
−
�
⋅
�
�
�
=
�
2
⋅
�
0
⋅
�
−
�
⋅
�
+
�
2
⋅
�
⋅
�
1
⋅
�
−
�
⋅
�
−
2
⋅
�
⋅
�
1
⋅
�
−
�
⋅
�
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=j 
0
​
 ⋅e 
−y⋅t
 +t⋅j 
1
​
 ⋅e 
−y⋅t
 +c
=−y⋅j 
0
​
 ⋅e 
−y⋅t
 −y⋅t⋅j 
1
​
 ⋅e 
−y⋅t
 +j 
1
​
 ⋅e 
−y⋅t
 
=y 
2
 ⋅j 
0
​
 ⋅e 
−y⋅t
 +y 
2
 ⋅t⋅j 
1
​
 ⋅e 
−y⋅t
 −2⋅y⋅j 
1
​
 ⋅e 
−y⋅t
 
​
 

Over Damped:
�
�
=
�
0
⋅
�
−
�
0
⋅
�
+
�
1
⋅
�
−
�
1
⋅
�
+
�
�
�
=
−
�
0
⋅
�
0
⋅
�
−
�
0
⋅
�
−
�
1
⋅
�
1
⋅
�
−
�
1
⋅
�
�
�
=
�
0
2
⋅
�
0
⋅
�
−
�
0
⋅
�
+
�
1
2
⋅
�
1
⋅
�
−
�
1
⋅
�
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=j 
0
​
 ⋅e 
−y 
0
​
 ⋅t
 +j 
1
​
 ⋅e 
−y 
1
​
 ⋅t
 +c
=−y 
0
​
 ⋅j 
0
​
 ⋅e 
−y 
0
​
 ⋅t
 −y 
1
​
 ⋅j 
1
​
 ⋅e 
−y 
1
​
 ⋅t
 
=y 
0
2
​
 ⋅j 
0
​
 ⋅e 
−y 
0
​
 ⋅t
 +y 
1
2
​
 ⋅j 
1
​
 ⋅e 
−y 
1
​
 ⋅t
 
​
 

What we did in the previous section was solve for the under damped case, but the other two cases require exactly the same style of derivation to get them working.

Solving the Critically Damped Case
Let's start with the easiest: the critically damped case. The first two unknowns 
�
c and 
�
y have exactly the same solution as in the under-damped case, 
�
=
�
+
�
⋅
�
�
c=g+ 
s
d⋅q
​
  and 
�
=
�
2
y= 
2
d
​
 , while the other two unknowns 
�
0
j 
0
​
  and 
�
1
j 
1
​
  can be found easily from the initial conditions 
�
0
x 
0
​
 , 
�
0
v 
0
​
 , and 
�
=
0
t=0:

�
0
=
�
0
⋅
�
−
�
⋅
0
+
0
⋅
�
1
⋅
�
−
�
⋅
0
+
�
�
0
=
�
0
+
�
x 
0
​
 
x 
0
​
 
​
  
=j 
0
​
 ⋅e 
−y⋅0
 +0⋅j 
1
​
 ⋅e 
−y⋅0
 +c
=j 
0
​
 +c
​
 

And for the velocity...

�
0
=
−
�
⋅
�
0
⋅
�
−
�
⋅
0
−
�
⋅
0
⋅
�
1
⋅
�
−
�
⋅
�
+
�
1
⋅
�
−
�
⋅
0
�
0
=
−
�
⋅
�
0
+
�
1
v 
0
​
 
v 
0
​
 
​
  
=−y⋅j 
0
​
 ⋅e 
−y⋅0
 −y⋅0⋅j 
1
​
 ⋅e 
−y⋅t
 +j 
1
​
 ⋅e 
−y⋅0
 
=−y⋅j 
0
​
 +j 
1
​
 
​
 

Giving us...

�
0
=
�
0
−
�
�
1
=
�
0
+
�
0
⋅
�
j 
0
​
 
j 
1
​
 
​
  
=x 
0
​
 −c
=v 
0
​
 +j 
0
​
 ⋅y
​
 

And that's it, easy!

Solving the Over Damped Case
The over-damped case is a little more difficult so let's first summarize some terms again to make our equations clearer.

�
0
=
�
0
⋅
�
−
�
0
⋅
�
�
1
=
�
1
⋅
�
−
�
1
⋅
�
E 
0
​
 
E 
1
​
 
​
  
=j 
0
​
 ⋅e 
−y 
0
​
 ⋅t
 
=j 
1
​
 ⋅e 
−y 
1
​
 ⋅t
 
​
 

Giving us...

�
�
=
�
0
+
�
1
+
�
�
�
=
−
�
0
⋅
�
0
−
�
1
⋅
�
1
�
�
=
�
0
2
⋅
�
0
+
�
1
2
⋅
�
1
x 
t
​
 
v 
t
​
 
a 
t
​
 
​
  
=E 
0
​
 +E 
1
​
 +c
=−y 
0
​
 ⋅E 
0
​
 −y 
1
​
 ⋅E 
1
​
 
=y 
0
2
​
 ⋅E 
0
​
 +y 
1
2
​
 ⋅E 
1
​
 
​
 

We'll start by finding the two unknowns 
�
0
y 
0
​
 , and 
�
1
y 
1
​
 . Just like before we are going to substitute these equations into our equation of motion, gathering up the coefficients for the exponential terms:

�
⋅
�
+
�
⋅
�
=
�
⋅
�
�
+
�
⋅
�
�
+
�
�
�
⋅
�
+
�
⋅
�
=
�
⋅
(
�
0
+
�
1
+
�
)
+
�
⋅
(
−
�
0
⋅
�
0
−
�
1
⋅
�
1
)
+
(
�
0
2
⋅
�
0
+
�
1
2
⋅
�
1
)
�
⋅
�
+
�
⋅
�
=
�
⋅
�
0
+
�
⋅
�
1
+
�
⋅
�
−
�
⋅
�
0
⋅
�
0
−
�
⋅
�
1
⋅
�
1
+
�
0
2
⋅
�
0
+
�
1
2
⋅
�
1
�
⋅
�
+
�
⋅
�
−
�
⋅
�
=
(
�
−
�
⋅
�
0
+
�
0
2
)
⋅
�
0
+
(
�
−
�
⋅
�
1
+
�
1
2
)
⋅
�
1
s⋅g+d⋅q
s⋅g+d⋅q
s⋅g+d⋅q
s⋅g+d⋅q−s⋅c
​
  
=s⋅x 
t
​
 +d⋅v 
t
​
 +a 
t
​
 
=s⋅(E 
0
​
 +E 
1
​
 +c)+d⋅(−y 
0
​
 ⋅E 
0
​
 −y 
1
​
 ⋅E 
1
​
 )+(y 
0
2
​
 ⋅E 
0
​
 +y 
1
2
​
 ⋅E 
1
​
 )
=s⋅E 
0
​
 +s⋅E 
1
​
 +s⋅c−d⋅y 
0
​
 ⋅E 
0
​
 −d⋅y 
1
​
 ⋅E 
1
​
 +y 
0
2
​
 ⋅E 
0
​
 +y 
1
2
​
 ⋅E 
1
​
 
=(s−d⋅y 
0
​
 +y 
0
2
​
 )⋅E 
0
​
 +(s−d⋅y 
1
​
 +y 
1
2
​
 )⋅E 
1
​
 
​
 

Again, just like before, this equation is only solved when both coefficients and the left hand side equal zero. This gives us our existing solution for 
�
c plus two more quadratic equations to solve:

�
−
�
⋅
�
0
+
�
0
2
=
0
�
−
�
⋅
�
1
+
�
1
2
=
0
s−d⋅y 
0
​
 +y 
0
2
​
 
s−d⋅y 
1
​
 +y 
1
2
​
 
​
  
=0
=0
​
 

In this case 
�
0
y 
0
​
  and 
�
1
y 
1
​
  represent the two different solutions to the same quadratic, so we can use the quadratic formula to get these:

�
0
=
�
+
�
2
−
4
⋅
�
2
�
1
=
�
−
�
2
−
4
⋅
�
2
y 
0
​
 
y 
1
​
 
​
  
= 
2
d+ 
d 
2
 −4⋅s
​
 
​
 
= 
2
d− 
d 
2
 −4⋅s
​
 
​
 
​
 

Now that these are found we're ready to solve for 
�
0
j 
0
​
  and 
�
1
j 
1
​
  using the initial conditions 
�
0
x 
0
​
 , 
�
0
v 
0
​
 , and 
�
=
0
t=0:

�
0
=
�
0
⋅
�
−
�
0
⋅
0
+
�
1
⋅
�
−
�
1
⋅
0
+
�
�
0
=
�
0
+
�
1
+
�
�
0
=
−
�
0
⋅
�
0
⋅
�
−
�
0
⋅
0
−
�
1
⋅
�
1
⋅
�
−
�
1
⋅
0
�
0
=
−
�
0
⋅
�
0
−
�
1
⋅
�
1
x 
0
​
 
x 
0
​
 
v 
0
​
 
v 
0
​
 
​
  
=j 
0
​
 ⋅e 
−y 
0
​
 ⋅0
 +j 
1
​
 ⋅e 
−y 
1
​
 ⋅0
 +c
=j 
0
​
 +j 
1
​
 +c
=−y 
0
​
 ⋅j 
0
​
 ⋅e 
−y 
0
​
 ⋅0
 −y 
1
​
 ⋅j 
1
​
 ⋅e 
−y 
1
​
 ⋅0
 
=−y 
0
​
 ⋅j 
0
​
 −y 
1
​
 ⋅j 
1
​
 
​
 

First we'll re-arrange our eqiation for 
�
0
x 
0
​
  in terms of 
�
0
j 
0
​
 

�
0
=
�
0
+
�
1
+
�
�
0
−
�
1
−
�
=
�
0
x 
0
​
 
x 
0
​
 −j 
1
​
 −c
​
  
=j 
0
​
 +j 
1
​
 +c
=j 
0
​
 
​
 

Which we'll substitute into our equation for 
�
0
v 
0
​
  to solve for 
�
1
j 
1
​
 :

�
0
=
−
�
0
⋅
�
0
−
�
1
⋅
�
1
�
0
=
−
�
0
⋅
(
�
0
−
�
1
−
�
)
−
�
1
⋅
�
1
�
0
=
−
�
0
⋅
�
0
+
�
0
⋅
�
1
+
�
0
⋅
�
−
�
1
⋅
�
1
−
�
0
=
�
0
⋅
�
0
−
�
0
⋅
�
1
−
�
0
⋅
�
+
�
1
⋅
�
1
�
0
⋅
�
−
�
0
−
�
0
⋅
�
0
=
−
�
0
⋅
�
1
+
�
1
⋅
�
1
�
0
⋅
�
−
�
0
−
�
0
⋅
�
0
=
�
1
⋅
(
�
1
−
�
0
)
�
0
⋅
�
−
�
0
−
�
0
⋅
�
0
�
1
−
�
0
=
�
1
v 
0
​
 
v 
0
​
 
v 
0
​
 
−v 
0
​
 
y 
0
​
 ⋅c−v 
0
​
 −y 
0
​
 ⋅x 
0
​
 
y 
0
​
 ⋅c−v 
0
​
 −y 
0
​
 ⋅x 
0
​
 
y 
1
​
 −y 
0
​
 
y 
0
​
 ⋅c−v 
0
​
 −y 
0
​
 ⋅x 
0
​
 
​
 
​
  
=−y 
0
​
 ⋅j 
0
​
 −y 
1
​
 ⋅j 
1
​
 
=−y 
0
​
 ⋅(x 
0
​
 −j 
1
​
 −c)−y 
1
​
 ⋅j 
1
​
 
=−y 
0
​
 ⋅x 
0
​
 +y 
0
​
 ⋅j 
1
​
 +y 
0
​
 ⋅c−y 
1
​
 ⋅j 
1
​
 
=y 
0
​
 ⋅x 
0
​
 −y 
0
​
 ⋅j 
1
​
 −y 
0
​
 ⋅c+y 
1
​
 ⋅j 
1
​
 
=−y 
0
​
 ⋅j 
1
​
 +y 
1
​
 ⋅j 
1
​
 
=j 
1
​
 ⋅(y 
1
​
 −y 
0
​
 )
=j 
1
​
 
​
 

After which is is easy to get 
�
0
j 
0
​
 :

�
0
=
�
0
+
�
1
+
�
�
0
−
�
1
−
�
=
�
0
x 
0
​
 
x 
0
​
 −j 
1
​
 −c
​
  
=j 
0
​
 +j 
1
​
 +c
=j 
0
​
 
​
 

And that's it! Let's add all of these different cases to our spring damper function:

void spring_damper_exact(
    float& x, 
    float& v, 
    float x_goal, 
    float v_goal, 
    float stiffness, 
    float damping, 
    float dt, 
    float eps = 1e-5f)
{
    float g = x_goal;
    float q = v_goal;
    float s = stiffness;
    float d = damping;
    float c = g + (d*q) / (s + eps);
    float y = d / 2.0f; 
    
    if (fabs(s - (d*d) / 4.0f) < eps) // Critically Damped
    {
        float j0 = x - c;
        float j1 = v + j0*y;
        
        float eydt = fast_negexp(y*dt);
        
        x = j0*eydt + dt*j1*eydt + c;
        v = -y*j0*eydt - y*dt*j1*eydt + j1*eydt;
    }
    else if (s - (d*d) / 4.0f > 0.0) // Under Damped
    {
        float w = sqrtf(s - (d*d)/4.0f);
        float j = sqrtf(squaref(v + y*(x - c)) / (w*w + eps) + squaref(x - c));
        float p = fast_atan((v + (x - c) * y) / (-(x - c)*w + eps));
        
        j = (x - c) > 0.0f ? j : -j;
        
        float eydt = fast_negexp(y*dt);
        
        x = j*eydt*cosf(w*dt + p) + c;
        v = -y*j*eydt*cosf(w*dt + p) - w*j*eydt*sinf(w*dt + p);
    }
    else if (s - (d*d) / 4.0f < 0.0) // Over Damped
    {
        float y0 = (d + sqrtf(d*d - 4*s)) / 2.0f;
        float y1 = (d - sqrtf(d*d - 4*s)) / 2.0f;
        float j1 = (c*y0 - x*y0 - v) / (y1 - y0);
        float j0 = x - j1 - c;
        
        float ey0dt = fast_negexp(y0*dt);
        float ey1dt = fast_negexp(y1*dt);

        x = j0*ey0dt + j1*ey1dt + c;
        v = -y0*j0*ey0dt - y1*j1*ey1dt;
    }
}
Awesome! Now it works even for very high damping values!


The Half-life and the Frequency
We're almost there, but our functions still use these mysterious damping and stiffness parameters. Can we turn these into something a bit more meaningful? Yes! Just like before we can use a halflife parameter instead of a damping parameter by controlling what we give as input to the exp functions:

float halflife_to_damping(float halflife, float eps = 1e-5f)
{
    return (4.0f * 0.69314718056f) / (halflife + eps);
}
    
float damping_to_halflife(float damping, float eps = 1e-5f)
{
    return (4.0f * 0.69314718056f) / (damping + eps);
}
Here as well as our previous constant of 
�
�
(
2
)
ln(2) we need to multiply by 4. This is a bit of a fudge factor but roughly it corresponds to the fact that we divide by two once to get 
�
y from 
�
d, and that the spring equation is usually a sum of two exponential terms instead of the single one we had for the damper.

What about the stiffness parameter? Well this one we can turn into a parameter called frequency:

float frequency_to_stiffness(float frequency)
{
   return squaref(2.0f * M_PI * frequency);
}

float stiffness_to_frequency(float stiffness)
{
    return sqrtf(stiffness) / (2.0f * M_PI);
}
Which is close to what will become 
�
w in the under-damped case.

Both are not completely honest names - the velocity continuity and oscillations of the spring means that the position will not be exactly half way toward the goal in halflife time, while the frequency parameter is more like a pseudo-frequency as the rate of oscillation is also affected by the damping value. Nonetheless, both give more intuitive controls for the spring than the damping and stiffness alternatives.


Along these lines, another useful set of functions are those that give us settings for these two parameters in the critical case (i.e. when 
�
2
4
=
�
4
d 
2
 
​
 =s ). These can be useful for setting defaults or in other cases when we only want to set one of these parameters.

float critical_halflife(float frequency)
{
    return damping_to_halflife(sqrtf(frequency_to_stiffness(frequency) * 4.0f));
}

float critical_frequency(float halflife)
{
    return stiffness_to_frequency(squaref(halflife_to_damping(halflife)) / 4.0f);
}
The critical_halflife function doesn't make that much sense since when critical there aren't any oscillations, but it can still be useful in certain cases. Putting it all together we can provide our spring with a nicer interface:

void spring_damper_exact(
    float& x, 
    float& v, 
    float x_goal, 
    float v_goal, 
    float frequency, 
    float halflife, 
    float dt, 
    float eps = 1e-5f)
{    
    float g = x_goal;
    float q = v_goal;
    float s = frequency_to_stiffness(frequency);
    float d = halflife_to_damping(halflife);
    float c = g + (d*q) / (s + eps);
    float y = d / 2.0f; 
    
    ...

}
Below you can see which critical frequency corresponds to a given halflife. And now I promise we really are done: that's it, an exact damped spring!


The Damping Ratio
Although controlling the frequency is nice, there is a different control which may be even better for users as it resembles a bit more the scale from less springy to more springy they might want. This is the damping ratio, where a value of 
1
1 means a critically damped spring, a value 
<
1
<1 means an under-damped spring, and a value 
>
1
>1 means a over-damped spring.

The equation for the damping ratio 
�
r is as follows, where 
�
d is the damping and 
�
s is the stiffness:

�
=
�
2
 
�
r= 
2  
s
​
 
d
​
 
​
 

This we can re-arrange to solve for stiffness or damping as required.

float damping_ratio_to_stiffness(float ratio, float damping)
{
    return squaref(damping / (ratio * 2.0f));
}

float damping_ratio_to_damping(float ratio, float stiffness)
{
    return ratio * 2.0f * sqrtf(stiffness);
}
And use instead of the frequency.

void spring_damper_exact_ratio(
    float& x, 
    float& v, 
    float x_goal, 
    float v_goal, 
    float damping_ratio, 
    float halflife, 
    float dt, 
    float eps = 1e-5f)
{    
    float g = x_goal;
    float q = v_goal;
    float d = halflife_to_damping(halflife);
    float s = damping_ratio_to_stiffness(damping_ratio, d);
    float c = g + (d*q) / (s + eps);
    float y = d / 2.0f; 
    
    ...

}
And here it is in action!


The Critical Spring Damper
Looking at our exact spring damper, the critical case is particularly interesting for us (and is most likely the case you may have actually used in your games) - not only because it's the situation where the spring moves toward the goal as fast as possible without additional oscillation, but because it's the easiest to compute and also to use as it has fewer parameters. We can therefore make a special function for it, which will allow us to remove the frequency parameter and throw in a few more basic optimizations:

void critical_spring_damper_exact(
    float& x, 
    float& v, 
    float x_goal, 
    float v_goal, 
    float halflife, 
    float dt)
{
    float g = x_goal;
    float q = v_goal;
    float d = halflife_to_damping(halflife);
    float c = g + (d*q) / ((d*d) / 4.0f);
    float y = d / 2.0f;	
    float j0 = x - c;
    float j1 = v + j0*y;
    float eydt = fast_negexp(y*dt);

    x = eydt*(j0 + j1*dt) + c;
    v = eydt*(v - j1*y*dt);
}
With no special cases for over-damping and under-damping this can compile down to something very fast. Separate functions for other common situations can be useful too, such as when the goal velocity q is zero...

void simple_spring_damper_exact(
    float& x, 
    float& v, 
    float x_goal, 
    float halflife, 
    float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    float j0 = x - x_goal;
    float j1 = v + j0*y;
    float eydt = fast_negexp(y*dt);

    x = eydt*(j0 + j1*dt) + x_goal;
    v = eydt*(v - j1*y*dt);
}
or when the goal position is zero too...

void decay_spring_damper_exact(
    float& x, 
    float& v, 
    float halflife, 
    float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    float j1 = v + x*y;
    float eydt = fast_negexp(y*dt);

    x = eydt*(x + j1*dt);
    v = eydt*(v - j1*y*dt);
}
Another optimization that can be useful is to pre-compute y and eydt for a given halflife. If there are many springs that need to be updated with the same halflife and dt this can provide a big speed-up.

Applications
Smoothing
Probably the most common application of springs in game development is smoothing - any noisy signal can be easily smoothed in real time by a spring damper and the half life can be used to control the amount of smoothing applied vs how responsive it is to changes.


Filtering
Springs also work well for filtering out sudden changes or jitters in signals, and even springs with quite a small halflife will do really well at removing any sudden jitters.


Controllers
Another common application of springs in game development is for moving characters. The usual process is to take the user input from the gamepad and turn it into a desired character velocity, which we then set as the goal for a spring. Each timestep we tick this spring and use what it produces as a velocity with which to move the character. By tweaking the parameters of the spring we can achieve movement with different levels of smoothness and responsiveness.

The slightly unintuitive thing to remember about this setup is that we are using the spring in a way such that its position corresponds to the character's velocity - meaning the spring's velocity will correspond to the character's acceleration.

Assuming the desired character velocity remains fixed we can also use this spring to predict the future character velocity by simply by evaluating the spring with a larger 
�
�
dt than we normally would and seeing what the result is.

If we want the character's position after some timestep (not just the velocity) we can compute it more accurately by using the the integral of our critical spring equation with respect to time. This will give us an accurate prediction of the future position of the character too.

�
�
=
∫
(
�
0
⋅
�
−
�
⋅
�
+
�
⋅
�
1
⋅
�
−
�
⋅
�
+
�
)
�
�
�
�
=
−
�
1
�
2
⋅
�
−
�
⋅
�
+
−
�
0
−
�
1
⋅
�
�
⋅
�
−
�
⋅
�
+
�
1
�
2
+
�
0
�
+
�
⋅
�
+
�
0
x 
t
​
 
x 
t
​
 
​
  
=∫(j 
0
​
 ⋅e 
−y⋅t
 +t⋅j 
1
​
 ⋅e 
−y⋅t
 +c)dt
= 
y 
2
 
−j 
1
​
 
​
 ⋅e 
−y⋅t
 + 
y
−j 
0
​
 −j 
1
​
 ⋅t
​
 ⋅e 
−y⋅t
 + 
y 
2
 
j 
1
​
 
​
 + 
y
j 
0
​
 
​
 +c⋅t+x 
0
​
 
​
 

And translated into code...

void spring_character_update(
    float& x, 
    float& v, 
    float& a, 
    float v_goal, 
    float halflife, 
    float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
    float j0 = v - v_goal;
    float j1 = a + j0*y;
    float eydt = fast_negexp(y*dt);

    x = eydt*(((-j1)/(y*y)) + ((-j0 - j1*dt)/y)) + 
        (j1/(y*y)) + j0/y + v_goal * dt + x;
    v = eydt*(j0 + j1*dt) + v_goal;
    a = eydt*(a - j1*y*dt);
}
This code is the same as the critically damped spring, but applied to the character velocity and the integral used to compute the character position. If we want to predict the future trajectory we can use this to update arrays of data each with a different 
�
�
dt:

void spring_character_predict(
    float px[], 
    float pv[], 
    float pa[], 
    int count,
    float x, 
    float v, 
    float a, 
    float v_goal, 
    float halflife,
    float dt)
{
    for (int i = 0; i < count; i++)
    {
        px[i] = x; 
        pv[i] = v; 
        pa[i] = a;
    }

    for (int i = 0; i < count; i++)
    {
        spring_character_update(px[i], pv[i], pa[i], v_goal, halflife, i * dt);
    }
}
This really shows how using a completely exact spring equation comes in handy - we can accurately predict the state of the spring at any arbitrary point in the future without having to simulate what happens in between.


Here you can see me moving around a point in the world using the above code and a desired character velocity coming from the gamepad. By adjusting the halflife of the spring we can achieve different levels of responsiveness and smoothness, and by evaluating the spring at various different times in the future we can predict where the character would be if the current input were to remain fixed (shown in red).

This is exactly the method we use to predict the future character trajectory in Learned Motion Matching.

Inertialization
In game animation, inertialization is the name given to a kind of blending that fades in or out an offset between two animations. Generally it can be use as a more performant alternative to a cross-fade blend since it only needs to evaluate one animation at a time. In the original presentation a polynomial is used blend out this offset smoothly, but springs can be used for this too.

The idea is this: if we have two different streams of animation we wish to switch between, at the point of transition we record the offset between the currently playing animation and the one we want to switch to. Then, we switch to this new animation but add back the previously recorded offset. We then decay this offset smoothly toward zero over time - in this case using a spring damper.

In code it looks something like this. First at the transition point we record the offset in terms of position and velocity between the currently playing animation src and the one we're going to switch to dst:

void inertialize_transition(
    float& off_x, float& off_v, 
    float src_x, float src_v,
    float dst_x, float dst_v)
{
    off_x = (src_x + off_x) - dst_x;
    off_v = (src_v + off_v) - dst_v;
}
We then switch to this animation, and at every frame we decay this offset toward zero, adding the result back to our currently playing animation.

void inertialize_update(
    float& out_x, float& out_v,
    float& off_x, float& off_v,
    float in_x, float in_v,
    float halflife,
    float dt)
{
    decay_spring_damper_exact(off_x, off_v, halflife, dt);
    out_x = in_x + off_x;
    out_v = in_v + off_v;
}
Here you can see it in action:


As you can see, each time we toggle the button there is a transition between the two different streams of animation (in this case two different 
sin
⁡
sin waves shown in red), while the inertialization smoothly fills in the difference (shown in blue).

Unlike the original presentation which uses a polynomial to blend out the offset over a specific period, a spring does not provide a fixed blend time and can easily overshoot. However the exponential decay does mean it tends to look smooth and blends out to something negligible at a very fast rate. In addition, since there is no need to remember the last transition time the code is very simple, and because we use the decay_spring_damper_exact variant of the spring it can be made exceptionally fast, in particular when all the blends for the different bones use the same halflife and dt to update.

This is exactly the method we use for switching between animations in our Motion Matching implementation as demonstrated in Learned Motion Matching.

Interpolation
The time dimension of a spring doesn't have to always be the real time that ticks by - it can be any variable that increases monotonically. For example we could use the parameterization along a curve as the time dimension to feed to a spring to produce a kind of spline.

Here I set up a piecewise interpolation of some 2D control points and used that as position and velocity goal for two springs (one for each dimension in 2D).

void piecewise_interpolation(
    float& x,
    float& v,
    float t,
    float pnts[],
    int npnts)
{
    t = t * (npnts - 1);
    int i0 = floorf(t);
    int i1 = i0 + 1;
    i0 = i0 > npnts - 1 ? npnts - 1 : i0;
    i1 = i1 > npnts - 1 ? npnts - 1 : i1;
    float alpha = fmod(t, 1.0f);
    
    x = lerp(pnts[i0], pnts[i1], alpha);
    v = (pnts[i0] - pnts[i1]) / npnts;
}
The result is a kind of funky spline which springs toward the control points. By adjusting the halflife and frequency we can produce some interesting shapes but overall the result has an odd feel to it since it's not symmetric and usually doesn't quite reach the last control point. Perhaps some more experimentation here could be interesting, such as running two springs along the control points in opposite directions and mixing the result.


I think there is probably still a way to formulate an interesting and useful type of spline using springs. Tell me if you manage to come up with anything interesting!

Resonance
If you've got a signal which you think contains at specific frequency but you don't know exactly which frequency it is what do you do? Well, I can already see you starting to type "fast fourier transform" into google but hold on a second, do you really need to do something that complicated?

Springs can be used to see if a signal is oscillating at a specific frequency! Fed with a goal moving at their resonate frequency springs will oscillate and build up energy, while fed a goal moving at any other frequency will make them die out and lose energy.

We can measure the energy of the spring using the sum of potential and kinematic energies.

float spring_energy(
    float x, 
    float v, 
    float frequency,
    float x_rest = 0.0f, 
    float v_rest = 0.0f,
    float scale = 1.0f)
{
    float s = frequency_to_stiffness(frequency);
    
    return (
        squaref(scale * (v - v_rest)) + s * 
        squaref(scale * (x - x_rest))) / 2.0f;
}
Then, if we want to see which frequency is contained in a signal we can simply drive a spring (or multiple springs at different frequencies) and see which setting produces the most energy.


As mentioned in the previous sections, the frequency parameter we're using doesn't actually reflect the true resonate frequency of the spring (the true resonate frequency is also affected by the damping), but we can find the frequency we need to set a spring to match some resonant frequency by fixing the halflife:

float resonant_frequency(float goal_frequency, float halflife)
{
    float d = halflife_to_damping(halflife);
    float goal_stiffness = frequency_to_stiffness(goal_frequency);
    float resonant_stiffness = goal_stiffness - (d*d)/4.0f;
    return stiffness_to_frequency(resonant_stiffness);
}
When trying to pick out specific frequencies the halflife of the spring affects the sensitivity. A long halflife (or low damping) means the spring will only build up energy when driven at frequencies very close to its resonate frequency, and it will build up more energy too. While a shorter halflife means a broader range of frequencies that build up energy.

For a really cool application of this idea check out this blog post by Kevin Bergamin.

Extrapolation
If we have an object moving at a velocity with damping applied we can use a damper to estimate a what the position might be at some time in the future.

The idea is to assume the velocity is being reduced over time via an exponential decay (i.e. via a damper):

�
�
=
�
0
⋅
�
−
�
⋅
�
v 
t
​
 
​
  
=v 
0
​
 ⋅e 
−y⋅t
 
​
 

Then, like in the controller example, we can then take the integral of this equation to work out the exact future position at some time 
�
t:

�
�
=
∫
�
0
⋅
�
−
�
⋅
�
�
�
�
�
=
�
0
�
 
(
1
−
�
−
�
⋅
�
)
+
�
0
x 
t
​
 
x 
t
​
 
​
  
=∫v 
0
​
 ⋅e 
−y⋅t
 dt
= 
y
v 
0
​
 
​
  (1−e 
−y⋅t
 )+x 
0
​
 
​
 

In code it looks something like this where halflife controls the decay rate of the velocity.

void extrapolate(
    float& x,
    float& v,
    float dt,
    float halflife,
    float eps = 1e-5f)
{
    float y = 0.69314718056f / (halflife + eps);
    x = x + (v / (y + eps)) * (1.0f - fast_negexp(y * dt));
    v = v * fast_negexp(y * dt);
}
And here is a little demo showing it in action:


Other Springs
Double Spring
You'll notice that the spring damper has a kind of asymmetric look to it - the start is very steep and it quickly evens out. We can achieve more of an "S" shape to the spring by using another spring on the goal. We'll deem this the "double spring":

void double_spring_damper_exact(
    float& x, 
    float& v, 
    float& xi,
    float& vi,
    float x_goal,
    float halflife, 
    float dt)
{
    simple_spring_damper_exact(xi, vi, x_goal, 0.5f * halflife, dt);
    simple_spring_damper_exact(x, v, xi, 0.5f * halflife, dt);
}
And here you can see it in action:


In red you can see the intermediate spring xi while in blue you can see the "double spring" which has a slightly more "S" shaped start and end.

Timed Spring
In some cases we might not want the spring to reach the goal immediately but at some specific time in the future - yet we might still want to keep the smoothing and filtering properties the spring brings when the goal changes quickly. Here is a spring variant that takes as input a goal time as well as a goal position and tries to achieve the goal at approximately the correct time. The idea is to track a linear interpolation directly toward the goal but to do so some time in the future (to give the spring time to blend out once it gets close to the goal).

void timed_spring_damper_exact(
    float& x,
    float& v,
    float& xi,
    float x_goal,
    float t_goal,
    float halflife,
    float dt,
    float apprehension = 2.0f)
{
    float min_time = t_goal > dt ? t_goal : dt;
    
    float v_goal = (x_goal - xi) / min_time;
    
    float t_goal_future = dt + apprehension * halflife;
    float x_goal_future = t_goal_future < t_goal ?
        xi + v_goal * t_goal_future : x_goal;
        
    simple_spring_damper_exact(x, v, x_goal_future, halflife, dt);
    
    xi += v_goal * dt;
}
Here the apprehension parameter controls how far into the future we try to track the linear interpolation. A value of 2 means two-times the half life, or that we expect the blend-out to be 75% done by the goal time. Below we can see this spring in action, with the linear interpolation toward the goal shown in red.


Velocity Spring
We can use a similar idea to make a spring that tries to maintain a given velocity by tracking an intermediate target that moves toward the goal at that fixed velocity.

void velocity_spring_damper_exact(
    float& x,
    float& v,
    float& xi,
    float x_goal,
    float v_goal,
    float halflife,
    float dt,
    float apprehension = 2.0f,
    float eps = 1e-5f)
{
    float x_diff = ((x_goal - xi) > 0.0f ? 1.0f : -1.0f) * v_goal;
    
    float t_goal_future = dt + apprehension * halflife;
    float x_goal_future = fabs(x_goal - xi) > t_goal_future * v_goal ?
        xi + x_diff * t_goal_future : x_goal;
    
    simple_spring_damper_exact(x, v, x_goal_future, halflife, dt);
    
    xi = fabs(x_goal - xi) > dt * v_goal ? xi + x_diff * dt : x_goal; 
}
And here is how this one looks, with the intermediate target shown in red.


Quaternion Spring
The simplified code of the simple spring damper also lends itself to be easily adapted to other things such as quaternions. Here the main trick is to convert quaternion differences into angular velocities (first convert to angle axis then scale the axis by the angle) so that they can interact with the other terms such as the exponential terms and the spring velocity.

void simple_spring_damper_exact_quat(
    quat& x, 
    vec3& v, 
    quat x_goal, 
    float halflife, 
    float dt)
{
    float y = halflife_to_damping(halflife) / 2.0f;	
	
    vec3 j0 = quat_to_scaled_angle_axis(quat_mul(x, quat_inv(x_goal)));
    vec3 j1 = v + j0*y;
	
    float eydt = fast_negexp(y*dt);

    x = quat_mul(quat_from_scaled_angle_axis(eydt*(j0 + j1*dt)), x_goal);
    v = eydt*(v - j1*y*dt);
}
One thing that is perhaps unintuitive about this derivation is the fact that we actually compute a rotation which takes the goal toward the initial state rather than the other way around (which is what we might expect).

It's a good exercise to try and do this same style of derivation for other spring variants and other quantities such as angles, but I'll leave that as an exercise for you...

Scale Spring
In this post I show how we can derive an equation for a spring that works on object scales.

Tracking Spring
See this article for a spring which can be used to perfectly track animation data while still removing discontinuities.

Conclusion
Source Code
The source code for all the demos shown in this article can be found here. They use raylib and more specifically raygui but once you have both of those installed you should be ready to roll.

Conclusion
I hope this article has piqued your interest in springs. Don't hesitate to get in contact if you come up with any other interesting derivations, applications, or spring variations. I'd be more than happy to add them to this article for others to see. Other than that there is not much more to say - happy exploring!

