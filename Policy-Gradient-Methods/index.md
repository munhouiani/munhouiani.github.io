---
title: "Policy Gradient Methods"
source: https://blog.munhou.com/Policy-Gradient-Methods/
date: 2023-03-01
updated: 2026-11-23
tags: [Reinforcement Learning]
---

# Policy Gradient Methods

## What Are Policy Gradient Methods

Policy gradient methods are a class of reinforcement learning algorithms used for *optimizing parametrized policies by gradient descent to maximize the expected return*, which is the cumulative reward received over a long-term horizon. These methods are based on updating the policy parameters such that the probabilities of actions leading to higher returns are increased. In comparison, the probabilities of actions leading to lower returns are decreased.

Iteratively optimizing the policy using this approach makes it possible to arrive at an optimal policy that maximizes the expected return. This technique has proven effective in solving complex decision-making problems in various fields, including robotics, game playing, and natural language processing.

## What is a Policy

In reinforcement learning, a policy, $\\pi$, is a function that maps an agent’s current observation to a distribution over actions that the agent can take. In the context of neural network-based policies, the policy function $\\pi\_{\\theta}$ is represented by a network with tunable parameters denoted by $\\theta$. The input to the policy network is typically a vector or matrix representing the agent’s observation. The output is a probability distribution over possible actions, with each action corresponding to a neuron in the output layer.

## Commonly Used Gradient Estimator

Given an expected return

$$  
\\hat{R} \_{\\theta} = \\sum \_{\\tau} R(t) \\pi \_{\\theta} (t)  
$$

Then the gradient

$$  
\\begin{aligned}  
\\nabla \\hat{R} \_{\\theta} &= \\hat{g} \\\\  
&= \\sum \_{\\tau} R( \\tau ) \\nabla \\pi \_{\\theta} (\\tau) \\\\  
&= \\sum \_{\\tau} R(\\tau) \\pi \_{\\theta} \\frac{\\nabla \\pi \_{\\theta} (\\tau)} {\\pi \_{\\theta}( \\tau)} \\\\  
&= \\sum \_{\\tau} R(\\tau) \\pi \_{\\theta} (\\tau) \\nabla \\log \\pi \_{\\theta} (\\tau) \\\\  
&= \\hat{ \\mathbb{E} } \_{\\tau \\sim \\pi \_{\\theta} (\\tau)} \\bigg\[ R(\\tau) \\nabla \\log \\pi \_{\\theta} (\\tau) \\bigg\] \\\\  
& \\approx \\frac{1}{N} \\sum ^{N} \_{n=1} R(\\tau^{n}) \\nabla \\log \\pi \_{\\theta}(\\tau ^n) \\\\  
&= \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} R(\\tau ^n) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
\\end{aligned}  
$$

where

-   $\\tau$ is a trajectory
-   $\\pi\_{\\theta}$ is a stochastic policy
-   $a^n\_t$ is the taken action of the $n$-th trajectory at time $t$
-   $s^n\_t$ is the state of the $n$-th trajectory at time $t$
-   $\\nabla\_{\\pi\_{\\theta}}$ is the gradient operator of the policy $\\pi$ with respect to $\\theta$
-   $\\hat{ \\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta}(\\tau)}\[\\dots\]$ is the empirical average expectation over a finite batch of samples in an algorithm that alternates between sampling and optimization
-   $\\frac{1}{N} \\sum^{N} \_{n=1} R(\\tau^{n}) \\nabla \\log \\pi \_{\\theta}(\\tau^n)$ is the approximation of the $\\hat {\\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta}(\\tau)}\[\\dots\]$ by sampling since it is impossible to collect all possible state and action pairs.

The $\\log$ term is a log-derivative trick that allows us to rewrite a derivative of a function as a product of its logarithm and its derivative and hence, simplify the computation of gradients for stochastic policies that are defined by probability distributions.

## Use Gradient Ascent to Update Policy

We use gradient ascent to update the parameters $\\theta$ of the policy $\\pi\_{\\theta}$ using policy gradient methods. The reason for this is that we want to encourage the taken action $\\pi\_{\\theta}(a^n\_t|s^n\_t)$ that results in a larger return $R(\\tau^n)$. The update rule for the parameters is given by:

$$  
\\theta \\leftarrow \\theta + \\eta \\nabla \\hat{R}\_{\\theta}  
$$

where

$$  
\\nabla \\hat{R} \_{\\theta} \\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} R(\\tau ^n) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
$$

The update rule nudges the parameters toward higher expected returns, improving the policy over time.

## Implementation Tip 1: Add a Baseline

Given that

$$  
\\nabla \\hat{R} \_{\\theta} \\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} R(\\tau^n) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
$$

We will subtract the return with a baseline $b$, as for every trajectory $\\tau^n$, the return $R(\\tau^n)$ may always be positive. Subtracting a baseline $b$ gives the equation a positive and negative value.

$$  
\\nabla \\hat{R} \_{\\theta} \\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} (R(\\tau^n) - b) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
$$

The selection of $b$ varies by choice. A straightforward implementation uses the expected value of the return $R(\\tau)$.

$$  
b \\approx \\mathbb{E}\[R(\\tau)\]  
$$

## Implementation Tip 2: Assign Suitable Credit

Given that

$$  
\\nabla \\hat{R} \_{\\theta} \\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} R(\\tau^n) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
$$

For the result of each specified state-action pair $\\pi\_{\\theta} (a^n\_t | s^n\_t)$, instead of weighting by the return of a trajectory $R(\\tau^n)$, it is more suitable to weight it by a discounted term, i.e., a discounted total reward after this state-action pair.

$$  
\\nabla \\hat{R} \_{\\theta} \\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum ^{T \_{n}} \_{t=1} \\sum ^{T\_n} \_{t’=t} \\gamma ^{t’ - t} r^n \_{t’} \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
$$

where  
$\\gamma$ is a discount factor, $\\gamma < 1$, usually set to $0.9$ or $0.99$

## Advantage Function

Combining implementation tip 1 and 2, we called $\\sum ^{T\_n} \_{t’=t} \\gamma ^{t’ - t} r^n \_{t’} - b$ an *advantage function*, denoted as

$$  
A^{\\theta}(s\_t, a\_t)  
$$

This advantage function can be estimated by a *“critic”*. The meaning of an advantage function is to tell how good it is if we take $a\_t$ other than other actions at $s\_t$

Therefore, the gradient term becomes

$$  
\\begin{aligned}  
\\nabla \\hat{R} \_{\\theta} &= \\hat{g} \\\\  
&= \\hat{\\mathbb{E}} \_{\\tau \\sim \\pi \_{\\theta}(\\tau)} \\bigg\[ A^{\\theta}(\\tau) \\nabla \\log \\pi \_{\\theta} (\\tau) \\bigg\] \\\\  
&\\approx \\frac{1}{N} \\sum^{N} \_{n=1} \\sum^{T \_{n}} \_{t=1} A ^{\\theta} (s^n\_t, a^n\_t) \\nabla \\log \\pi \_{\\theta} (a^n\_t | s^n\_t)  
\\end{aligned}  
$$
