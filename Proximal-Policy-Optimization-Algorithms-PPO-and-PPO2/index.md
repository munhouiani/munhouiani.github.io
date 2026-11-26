---
title: "Proximal Policy Optimization Algorithms (PPO and PPO2)"
source: https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/
date: 2023-03-01
updated: 2026-11-25
tags: [Reinforcement Learning]
---

# Proximal Policy Optimization Algorithms (PPO and PPO2)

Recall that from [Off-Policy Gradient Methods](https://blog.munhou.com/Off-Policy-Gradient-Methods/ "Off-Policy Gradient Methods"), the objective function is

$$  
\\begin{aligned}  
J^{\\theta’}(\\theta) &= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’} (\\tau)} \\bigg\[ \\frac{\\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta’}(a\_t | s\_t)} A^{\\theta’}(s\_t, a\_t) \\bigg\]  
\\end{aligned}  
$$

We learned that from [Importance Sampling](https://blog.munhou.com/Importance-Sampling/ "Importance Sampling"), the two distributions shouldn’t differ too much, therefore in PPO, they added a constraint term:

$$  
J ^{\\theta’} \_{PPO} (\\theta) = J ^{\\theta’} (\\theta) - \\beta KL(\\theta, \\theta’)  
$$

Where $\\beta$ is a weight, this constraint term ensures that $\\theta$ and $\\theta’$ are similar.

Before PPO, there was a method called TRPO (Trust Region Policy Optimization), and its objective function is

$$  
\\begin{aligned}  
J ^{\\theta’} \_{TRPO}(\\theta) &= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’}(\\tau)} \\bigg\[ \\frac{\\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta’}(a\_t | s\_t)} A ^{\\theta’}(s\_t, a\_t) \\bigg\] \\  
\\end{aligned}  
$$

with constrain $KL(\\theta, \\theta’) < \\delta$. We can see that PPO and TRPO are similar, but PPO puts the constraint term into its objective function. Therefore, in practice, PPO is easier to implement.

$KL(\\theta, \\theta’)$ is to ensure that the *behavior* (output) of $\\theta$ and $\\theta’$ are similar, but not their parameters (weights). Think $\\theta$ and $\\theta’$ as neural networks. Given a state $s\_t$, they both will output a distribution of actions. KL divergence is to ensure that these two distributions are similar.

Overall, the method is as follows:

1.  Initiate the policy parameters $\\theta^0$
    
2.  In each iteration,
    
    -   Use $\\theta^k$ to interact with the environment to collect ${s\_t, a\_t}$ and compute advantage $A^{\\theta^{k}}(s\_t, a\_t)$
    -   Find $\\theta$ optimizing $J\_{PPO}(\\theta)$
    
    $$\\begin{aligned}  
    J ^{\\theta^{k}} \_{PPO}(\\theta) &= J ^{\\theta ^{k}}(\\theta) - \\beta KL(\\theta, \\theta^{k}) \\\\  
    &= \\sum \_{(s\_t, a\_t)} \\frac{ \\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta^k} (a\_t | s\_t)} A ^{\\theta^{k}}(s\_t, a\_t) - \\beta KL(\\theta, \\theta ^{k})  
    \\end{aligned}  
    $$
    
    -   Pre-define maximum and minimum values for $KL$. If $KL(\\theta, \\theta^k) > KL\_{\\max}$, increase $\\beta$. Otherwise, decrease $\\beta$.

Since $KL(\\theta, \\theta’)$ could increase computation overheads, they also introduce PPO2, which the objective function is

$$  
\\begin{aligned}  
J ^{\\theta^{k}} \_{PPO2}(\\theta) \\approx \\sum \_{(s\_t, a\_t)} \\min \\bigg( & \\frac{\\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta^k} (a\_t | s\_t)} A^{\\theta^{k}}(s\_t, a\_t), \\\\  
& \\operatorname{clip} \\bigg( \\frac{\\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta^k} (a\_t | s\_t)}, 1 - \\epsilon, 1 + \\epsilon \\bigg) A ^{\\theta^{k}}(s\_t, a\_t) \\bigg)  
\\end{aligned}  
$$

This objective is easier to implement, and ensures that $\\theta$ doesn’t differ too much from $\\theta^k$ by:

-   When $A > 0$, means that the given $(s\_t, a\_t)$ is good. Therefore, we should optimize $\\theta$ by increasing the probability of $a$ given $s$. But, the difference between $\\theta$ and $\\theta^k$ is clipped by $1 + \\epsilon$. See the $\\min$ and $\\operatorname{clip}$.
-   Similarly, when $A < 0$, Therefore, we should optimize $\\theta$ by reducing the probability of $a$ given $s$. The difference between $\\theta$ and $\\theta^k$ is clipped by $1 - \\epsilon$. See the $\\min$ and $\\operatorname{clip}$.

In the PPO2 algorithm, we will no longer need to calculate the $KL$ while imposing the same constraint.
