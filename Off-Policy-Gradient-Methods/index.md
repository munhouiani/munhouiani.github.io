---
title: "Off-Policy Gradient Methods"
source: https://blog.munhou.com/Off-Policy-Gradient-Methods/
date: 2023-03-01
updated: 2026-11-25
tags: [Reinforcement Learning]
---

# Off-Policy Gradient Methods

Recall that from [Policy Gradient Methods](https://blog.munhou.com/Policy-Gradient-Methods/ "Policy Gradient Methods"), the calculation of gradient is

$$  
\\begin{aligned}  
\\nabla \\hat{R} \_{\\theta} &= \\hat{ \\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta}(\\tau)} \\bigg\[ R(\\tau) \\nabla \\log \\pi \_{\\theta} (\\tau) \\bigg\]  
\\end{aligned}  
$$

where

-   $\\tau$ is a trajectory
-   $\\pi\_{\\theta}$ is a stochastic policy
-   $\\nabla\_ {\\pi \_{\\theta}}$ is the gradient operator of the policy $\\pi$ with respect to $\\theta$
-   $\\hat{\\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta}(\\tau)}\[\\dots\]$ is the empirical average expectation over a finite batch of samples in an algorithm that alternates between sampling and optimization

In this setting, we have to:

-   Use $\\pi \_{\\theta}$ to collect data. When the parameter $\\theta$ is updated, we must sample the training data again.
-   This is called *on-policy* training

With [Importance Sampling](https://blog.munhou.com/Importance-Sampling/ "Importance Sampling"), we can alter the equation to

$$  
\\begin{aligned}  
&= \\hat{ \\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta’} (\\tau)} \\bigg\[ \\frac{ \\pi \_{\\theta}(\\tau)} {\\pi \_{\\theta’} (\\tau)} R(\\tau) \\nabla \\log \\pi \_{\\theta} (\\tau) \\bigg\]  
\\end{aligned}  
$$

In this setting, we can:

-   Use the samples drawn from a *fixed* distribution $\\pi\_{\\theta’}$ to train $\\theta$. ($\\theta’$ and $\\theta$ are different)
-   We can sample data once to train $\\theta$ many times.
-   To a certain extent, we can update $\\pi\_{\\theta’}$, sample new data, and use the new data to train $\\theta$.
-   This is called *off-policy* training.

We learned from [Policy Gradient Methods](https://blog.munhou.com/Policy-Gradient-Methods/ "Policy Gradient Methods") that we would use an advantage function $A^\\theta$. Therefore the gradient should be

$$  
\\begin{aligned}  
&= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’} (\\tau)} \\bigg\[ \\frac{ \\pi \_{\\theta} (s\_t, a\_t)}{\\pi \_{\\theta’}(s\_t, a\_t)} A^{\\theta’} (s\_t, a\_t) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t) \\bigg\] \\\\  
&= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’} (\\tau)} \\bigg\[ \\frac{ \\pi \_{\\theta} (a\_t | s\_t)} {\\pi \_{\\theta’}(a\_t | s\_t)} \\frac{\\pi \_{\\theta}(s\_t)}{\\pi \_{\\theta’} (s\_t)} A^{\\theta’} (s\_t, a\_t) \\nabla \\log \\pi \_{\\theta}(a^n\_t | s^n\_t) \\bigg\] \\\\  
&= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’}(\\tau)} \\bigg\[ \\frac{\\pi \_{\\theta} (a\_t | s\_t)}{\\pi \_{\\theta’} (a\_t | s\_t)} A^{\\theta’} (s\_t, a\_t) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t) \\bigg\] \\\\  
\\end{aligned}  
$$

Note that:

-   $(s\_t, a\_t)$ are training data drawn from a fix distribution $\\pi\_{\\theta’}$
-   The calculation of the advantage function $A^{\\theta’}$ is based on $\\theta’$
-   In line 2, we assume that $\\pi\_{\\theta}(s\_t)$ equals to $\\pi\_{\\theta’}(s\_t)$, therefore we eliminate $\\frac{\\pi\_{\\theta}(s\_t)}{\\pi\_{\\theta’}(s\_t)}$. This hypothesis assumes that the probability of running into a state $s\_t$ is independent from the $\\theta$ (environment).

Recall that for a gradient:

$$  
\\nabla f(x) = f(x) \\nabla \\log f(x)  
$$

Let $f(x) = \\pi\_{\\theta}(a^n\_t | s^n\_t)$. From above equation, combing $\\pi\_{\\theta}(a^n\_t | s^n\_t) \\nabla \\log \\pi\_{\\theta}(a^n\_t | s^n\_t)$ will get $\\nabla \\pi\_{\\theta}(a^n\_t | s^n\_t)$. Hence

$$  
\\begin{aligned}  
&= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’}(\\tau)} \\bigg\[ \\frac{ \\nabla \\pi \_{\\theta}(a\_t | s\_t)} {\\pi \_{\\theta’}(a\_t | s\_t)} A^{\\theta’}(s\_t, a\_t) \\bigg\]  
\\end{aligned}  
$$

By removing the gradient, we now have an objective function:

$$  
\\begin{aligned}  
J ^{\\theta’} (\\theta) &= \\hat{\\mathbb{E}} \_{(s\_t, a\_t) \\sim \\pi \_{\\theta’}(\\tau)}\\bigg\[ \\frac{\\pi \_{\\theta}(a\_t | s\_t)}{\\pi \_{\\theta’}(a\_t | s\_t)} A^{\\theta’}(s\_t, a\_t) \\bigg\] \\  
\\end{aligned}  
$$

This means we use data sampled from $\\theta’$ to update $\\theta$.
