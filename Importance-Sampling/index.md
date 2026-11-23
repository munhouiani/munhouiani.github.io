---
title: "Importance Sampling"
source: https://blog.munhou.com/Importance-Sampling/
date: 2023-03-01
updated: 2026-11-23
---

# Importance Sampling

Importance sampling is a technique commonly used in statistics and machine learning to estimate properties of a distribution that may be difficult to compute directly.

Suppose we want to calculate the expectation of a distribution $p$, but we don’t have a closed-form expression for it. Instead, we can approximate it by drawing samples from $p$ and using the average of these samples. Specifically, we draw $N$ samples from $p$ denoted as $x^i$, where $i$ ranges from $1$ to $N$. Then, we compute the average value of function $f(x)$ over these samples. Mathematically, the approximation is expressed as:

$$  
E\_{x \\sim p}\\big\[ f(x) \\big\] \\approx \\frac{1}{N} \\sum^{N}\_{i=1} f(x^i)  
$$

Drawing samples directly from the distribution $p$ can sometimes be difficult or computationally expensive. Importance sampling provides a way to estimate the expectation of a function $f(x)$ with respect to $p$ by instead drawing samples from a different distribution $q$. The basic idea is to use the samples from $q$ to estimate the integral of $f(x)$ with respect to $p$ by applying a weighting factor. Specifically, we can express the expectation of $f(x)$ with respect to $p$ as:

$$  
\\begin{aligned}  
E\_{x \\sim p}\\big\[ f(x) \\big\] &= \\int f(x)p(x) dx \\\\  
&= \\int f(x) \\frac{p(x)}{q(x)} q(x) dx \\\\  
&= E\_{x \\sim q}\\bigg\[ f(x) \\frac{p(x)}{q(x)} \\bigg\] \\\\  
&\\approx \\frac{1}{N} \\sum^{N}\_{i=1} f(x^i) \\frac{p(x^i)}{q(x^i)} \\\\  
\\end{aligned}  
$$

where $x^i$ is a sample drawn from $q$, and the importance weight $\\frac{p(x^i)}{q(x^i)}$ is used to adjust for the difference between the two distributions. Using this weighting factor, we can generate more accurate estimates of the expectation of $f(x)$ with respect to $p$ using samples drawn from $q$, even if drawing samples directly from $p$ is difficult or impractical.

By the way, this weight term assigns high weight to samples that are more likely to contribute to the expected value, and low weight to samples that are less likely to contribute, hence the name, “Importance Sampling”.

## Issue of Importance Sampling

Importance sampling aims to estimate the expectation of a target distribution $p$ by drawing samples from another distribution $q$. However, it is important to note that the two distributions cannot differ too much. This is because the weight $\\frac{p(x)}{q(x)}$ used to account for the difference between $p$ and $q$ can be very large if the difference between the two distributions is significant. As a result, the variance of the estimator can be large, which may require a larger number of samples to obtain an accurate estimate. To illustrate this, we can compare the variances of $\\operatorname{VAR} \_{x \\sim p} \[f(x)\]$ and $\\operatorname{VAR} \_{x \\sim q} \[f(x) \\frac{p(x)}{q(x)}\]$.

Recall that the equation of variance is

$$  
\\operatorname{VAR} \\big\[ X \\big\] = E \\big\[ X^2 \\big\] - (E \\big\[ X \\big\])^2  
$$

Hence:

$$  
\\begin{aligned}  
\\operatorname{VAR} \_{x \\sim p} \\big\[ f(x) \\big\] &= E \_{x \\sim p} \\big\[ f(x)^2 \\big\] - \\big( E \_{x \\sim p} \\big\[ f(x) \\big\] \\big)^2  
\\end{aligned}  
$$

and

$$  
\\begin{aligned}  
\\operatorname{VAR} \_{x \\sim q} \\bigg\[ f(x) \\frac{p(x)}{q(x)} \\bigg\] &= E \_{x \\sim q} \\bigg\[ \\bigg( f(x) \\frac{p(x)}{q(x)} \\bigg) ^2 \\bigg\] - E \_{x \\sim q} \\bigg\[ \\bigg( f(x) \\frac{p(x)}{q(x)} \\bigg) \\bigg\]^2 \\\\  
&= E \_{x \\sim p} \\bigg\[ \\bigg( f(x) ^2 \\frac{p(x)}{q(x)} \\bigg) \\bigg\] - \\big( E \_{x \\sim p} \\big\[ f(x) \\big\] \\big) ^2  
\\end{aligned}  
$$

Note that the difference between the two variances depends on the weight $\\frac{p(x)}{q(x)}$, which can be large if $p$ and $q$ differ significantly. Therefore, choosing a suitable distribution $q$ close enough to $p$ for importance sampling to be effective is crucial.
