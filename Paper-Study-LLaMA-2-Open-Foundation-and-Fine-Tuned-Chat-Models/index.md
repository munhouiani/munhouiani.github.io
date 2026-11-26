---
title: "Paper Study: LLaMA 2 Open Foundation and Fine-Tuned Chat Models"
source: https://blog.munhou.com/Paper-Study-LLaMA-2-Open-Foundation-and-Fine-Tuned-Chat-Models/
date: 2023-08-11
updated: 2026-11-25
tags: [Deep Learning, Large Language Model, Paper Study]
---

# LLaMA 2 Open Foundation and Fine-Tuned Chat Models

## 📖 Abstract

In this work, we develop and release Llama 2, a collection of pretrained and ﬁne-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our ﬁne-tuned LLMs, called LLaMA 2-Chat are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closedsource models. We provide a detailed description of our approach to ﬁne-tuning and safety improvements of LLaMA 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

## 🎯 Goal

-   Develop and release LLaMA 2, a pre-trained foundation language model, and a fine-tuned language model, LLaMA 2-Chat

## 🤔 Hypothesis

-   With sufficient scale, computing, and techniques to align with human preferences, it is possible to create large language models comparable in quality to closed-source chatbots through an open research process.

## 💡 Methodology

### Pre-Training

-   Using a similar pre-training method as LlaMA 1, except for the following differences:

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at214248.png)

#### Pre-training Data

-   They used a new mix of data from publicly available sources,  including web pages, books, Wikipedia, forums, and other text-based sites, but didn’t mention the exact sources.
-   They excluded data from certain sites known to contain high volumes of personal information about private individuals.
-   The training corpus has 2 trillion tokens.
-   The majority of the data is in English:

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at215358.png)

-   The data has a cutoff of September 2022.
-   The data was collected adhering to standard privacy and legal reviews and includes no private Meta user data.

#### Model Architecture

-   Adopt most of the pretraining setting from Llama1
    -   Standard Transformer architecture
    -   Apply RMSNorm
    -   Use SwiGLU activation function
    -   Use RoPE
-   Increase context length to **4096** tokens
-   Use grouped-query attention (GQA)

#### Hyperparemeters

-   Used AdamW optimizer
    -   $\\beta\_1 = 0.9$
    -   $\\beta\_2 = 0.95$
    -   $\\epsilon = 10^{-5}$
    -   Cosine learning rate schedule, with a warmup of 2000 steps, and decay final learning rate down to 10% of the peak learning rate.
    -   Use weight decay of 0.1
    -   Use gradient clipping of 1.0

#### Tokenizer

-   Same tokenizer as Llama1
-   use [BPE](https://blog.munhou.com/Byte-Pair-Encoding-BPE/ "Byte-Pair-Encoding (BPE)")
-   The total vocabulary size is 32K tokens.

### Fine-Tuned For Chat

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at211039.png)

#### Supervised Fine-Tuning (SFT)

##### SFT Data

-   Annotated 27,540 supervised fine-tuning datasets.
-   SFT annotations in the order of **tens of thousands** were enough to achieve a high-quality result.
-   Each sample consists of a prompt and an answer.
-   To annotate samples for safety, they asked human annotators to create adversarial prompts along two dimensions: a *risk category*, or potential topic about which the LLM could produce unsafe content, and an *attack vector*, or question style to cover different varieties of prompts that could elicit bad model behaviors.
    -   Three risk categories:
        1.  *Illicit and criminal activities* (e.g., terrorism, theft, human trafficking)
        2.  *Hateful and harmful activities* (e.g., defamation, self-harm, eating disorders, discrimination)
        3.  *Unqualified advice* (e.g., medical advice, financial advice, legal advice)
    -   Example of attack vector:
        -   *psychological manipulation* (e.g., authority manipulation)
        -   *logic manipulation* (e.g., false premises)
        -   *syntactic manipulation* (e.g., misspelling)
        -   *semantic manipulation* (e.g., metaphor)
        -   *perspective manipulation* (e.g., role-playing)
        -   *non-English languages*
-   They defined best practices for safe and helpful model responses:
    1.  The model should ﬁrst address immediate safety concerns if applicable.
    2.  Then, address the prompt by explaining the potential risks to the user.
    3.  Finally, provide additional information if possible.
-   They observed that a few thousand safety-related samples were enough for SFT.
-   Example of SFT annotation:

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at225631.png)

##### SFT Hyperparemeters

-   Use cosine learning rate schedule
-   Use an initial learning rate of $2 \\times 10^{-5}$
-   Use weight decay of 0.1
-   Use a batch size of 64
-   The sequence length is 4096 tokens.
-   2 epochs.

##### SFT Details

-   Concatenate all the prompts and answers from the training set to ensure the model sequence length is properly filled.
-   Use a special token to separate the prompt and answer segments.
-   Use an autoregressive objective.
-   Zero out the loss on tokens from the user prompt, only back-propagate on answer tokens.

#### Reinforcement Learning with Human Feedback (RLHF)

##### RLHF Data

-   Ask annotators to write a prompt first, then choose between two sampled model responses based on the provided criteria.
    -   Two responses are sampled from two different model variants and varying the temperature hyper-parameter.
-   Asked annotators also to label the degree to which they prefer their chosen response over the alternative: *significantly better*, *better*, *slightly better*, or *negligibly better / unsure*.
-   Asked annotators to label model responses into one of three categories: *1) the preferred response is safe and the other response is not*, *2) both responses are safe*, and *3) both responses are unsafe*. They ***didn’t*** use samples where the chosen response was unsafe and the other was safe.
-   These data are used to train two reward models: *Helpfulness RM*, optimized for helpfulness, and *Safety RM*, optimized for safety.
-   Human annotations were collected in batches on a weekly basis.
-   They collected over ***1 million*** binary comparisons by humans and have longer, more conversation turns. See the comparison against open datasets:

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at235927.png)

##### RLHF Reward Modeling

-   Trained two reward models: **Helpfulness RM**, optimized for helpfulness, and **Safety RM**, optimized for safety.
-   They initialized both reward models from pre-trained chat model checkpoints to ensure that both models benefit from the knowledge acquired in pre-training.
-   Model architecture and hyper-parameters are identical to the pre-trained language models, except that the classification head for next-token prediction is replaced with a regression head for outputting a scalar reward.
-   The training objective is binary ranking loss, enforcing the chosen response to score higher than its counterpart:  
    $$  
    \\operatorname{Loss\_{\\text{ranking}}} = -\\log(\\sigma(r\_{\\theta}(x, y\_c) - r\_{\\theta}(x, y\_r) - m(r)))  
    $$
-   Where $r\_{\\theta}(x, y)$ is the scalar score output for prompt $x$ and completion $y$ with model weights $\\theta$. $y\_c$ is the preferred response and $y\_r$ is the rejected counterpart. The margin $m(r)$ is a discrete function of the preference rating.
-   They use a large margin for pairs with distinct responses and a smaller one for those with similar responses. There are two variants for setting the margin function. See the table below.

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-10at001305.png)

-   The final training data composition for both reward models:
    -   **Helpfulness RM**:
        -   All Meta Helpfulness data
        -   Equal parts of the remaining data were uniformly sampled from Meta Safety and the open-source datasets.
    -   **Safety RM**:
        -   All Meta Safety data
        -   All Anthropic Harmless data
        -   Mixed with Meta Helpfulness and open-source helpfulness data in a 90/10 proportion.

##### RLHF Reward Model Training Details

-   Train for one epoch over the training data
-   Use the same optimizer parameters as for the base model:
    -   Used AdamW optimizer
        -   $\\beta\_1 = 0.9$
        -   $\\beta\_2 = 0.95$
        -   $\\epsilon = 10^{-5}$
        -   Cosine learning rate schedule, with a warmup step of 3% of the total number of steps, with a minimum of 5 warmup steps, and decay final learning rate down to 10% of the peak learning rate.
        -   Use weight decay of 0.1
        -   Use gradient clipping of 1.0
-   The maximum learning rate is $5 \\times 10^{-6}$ for the 70B parameter LLaMA 2-Chat, and $1 \\times 10^{-5}$ for the rest.
-   The batch size is kept fixed at 512 pairs, or 1024 rows per batch.

##### RLHF Iterative Fine-Tuning

-   They trained 5 successive versions for RLHF models, referred to here as *RLHF-V1*, *RLHF-V2*, *RLHF-V3*, *RLHF-V4*, *RLHF-V5*
-   They fine-tuned the models with:
    -   [Proximal Policy Optimization Algorithms](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") ([PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)"))
    -   Rejection Sampling fine-tuning
        -   The model samples K candidate responses for each prompt in the RLHF dataset.
        -   The reward models score these *K* responses.
        -   The response with the highest reward score is selected as the “best” response for that prompt.
        -   The prompts and their corresponding best responses selected via rejection sampling comprise a new training set.
        -   The model is then trained to reinforce the responses that receive high reward scores.
-   Rejection sampling is only done with the 70B LLaMA 2-Chat. All smaller models are fine-tuned on rejection sampled data from the larger model.
-   The new training set in each iteration includes the rejection-sampled data from the current iteration and top-performing samples from all prior iterations.
-   Until *RLHF-V4*, they used only Rejection Sampling fine-tuning. After that, they first applied Rejection Sampling, then applied [PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") on top of the resulting Rejection Sampling checkpoint before sampling again.
-   The reward function they use in the [PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") is:

$$  
R(g | p) = \\tilde{R\_c}(g|p) - \\beta D\_{KL}(\\pi\_{\\theta}(g|p) || \\pi\_0(g|p))  
$$

-   Where $\\pi\_0$ is the original policy. $R\_c$ is the piecewise combination of safety ($R\_s$) and helpfulness ($R\_h$) reward models:

$$  
\\begin{align}  
R\_c(g|p) &=  
\\begin{cases}  
R\_s(g|p) & \\text{if $\\operatorname{IS\\\_SAFETY}(p)$ or $R\_s(g|p) < 0.15$} \\\\  
R\_h(g|p) & \\text{otherwise}  
\\end{cases} \\\\  
\\tilde{R\_c}(g|p) &= \\operatorname{WHITEN}(\\operatorname{LOGIT}(R\_c(g|p)))  
\\end{align}  
$$

##### RLHF Iterative Fine-Tuning Training Details

-   Used AdamW optimizer
    -   $\\beta\_1 = 0.9$
    -   $\\beta\_2 = 0.95$
    -   $\\epsilon = 10^{-5}$
    -   Use a constant learning rate of $10^{-6}$
    -   Use weight decay of 0.1
    -   Use gradient clipping of 1.0
-   For each [PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") operation, they:
    -   Use a batch size of 512
    -   Use a [PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") clip threshold of 0.2
    -   Use a mini-batch size of 64
    -   Take 1 gradient step per mini-batch
    -   They set the $\\beta = 0.01$ (KL penalty) for 7B and 13B models, and $\\beta = 0.005$ for the 34B and 70B models.
-   They trained all the models for between 200 and 400 iterations and used evaluations on held-out prompts for early stopping.

#### Ghost Attention for Multi-Turn Consistency

-   They observed that LLaMA-2 Chat tended to forget the initial instruction (e.g., system prompt) after a few turns of dialog.
-   They introduced Ghost Attention (GAtt) to solve this problem.
-   GAtt augments the training data to “hack” the model’s attention mechanism.
-   Key attributes or instructions that should persist across a multi-turn conversation are concatenated to every user turn in the conversation during training.
-   For example, an instruction like “Act as a doctor” would be appended to each user utterance.
-   This forces the model to attend to those attributes when generating each assistant response.
-   However, when fine-tuning the model, the loss is set to 0 for those appended tokens, so they are not predicted.
-   So the attributes/instructions are like “ghosts” during training - not predicted but forced to be attended to.
-   This causes the model to learn to focus attention on those key attributes throughout the conversation, even when they are no longer appended at inference time.
-   As a result, the model is more consistent about following the instructions or maintaining persona/character throughout multiple turns.

## 📊 Result(s)

### Pre-Training

#### Training Loss

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at222311.png)

#### Benchmarks

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at222726.png) ![](https://blog.munhou.com/images/llama2/CleanShot2023-08-09at222758.png)

### Fine-Tuned For Chat

#### Reward Model Results

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-10at231055.png) ![](https://blog.munhou.com/images/llama2/CleanShot2023-08-10at231154.png) ![](https://blog.munhou.com/images/llama2/CleanShot2023-08-10at231451.png)

#### Evolution of Harmlessness

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-11at233511.png)

#### Human Evaluation

![](https://blog.munhou.com/images/llama2/CleanShot2023-08-11at233615.png)

## 🔑 Summary of Key Points

-   Pretrain large auto-regressive language models LLaMA 2 using publicly available data at scales up to 70B parameters, with optimizations like longer context and grouped query attention
-   Apply supervised fine-tuning (SFT) on high-quality human annotations for initializing helpful and safe models
-   Use reinforcement learning with human feedback (RLHF) to iteratively improve models by learning from human preferences
-   Propose techniques like rejection sampling, targeted context distillation, and Ghost Attention to enhance multi-turn dialogue capabilities
-   Conduct detailed investigations into pretraining data distributions and model capabilities to enable transparency
-   Share analysis of carbon emissions from pretraining and steps taken to increase efficiency and offset costs
-   Demonstrate with human evaluations that Llama 2-Chat models are comparable to closed-source chatbots like ChatGPT
-   Discuss techniques to mitigate risks identified through red teaming and limitations of current benchmark evaluations
-   Release Llama 2 openly, along with documentation to encourage reproducible research into aligning large language models
-   Argue that an open research approach with sufficient resources can produce models rivaling closed-source systems, supporting the decentralization of AI capabilities

## ❗ Significance

-   Demonstrates the viability of an open research approach to developing helpful and safe conversational AI systems comparable to closed-source chatbots.
-   Provides extensive documentation of techniques, data, and findings to enable transparency and reproducibility of aligning large language models.
-   Shares detailed analysis and mitigation strategies for safety risks in dialogue systems identified through red teaming.
-   Introduces optimizations like grouped query attention and Ghost Attention that improve scalability and multi-turn consistency in large models.
-   Analyzes trends in scaling model size, data, and computing to guide efficient training of future systems.
-   Discusses techniques to balance safety with open-domain performance through targeted data collection and fine-tuning.
-   Encourages responsible and transparent release of foundational AI models to increase access and allow decentralized innovation.
-   Reveals interesting observations about emergent behaviors like tool usage and temporal reasoning in aligned models.
-   Provides benchmarks and insights to advance research into evaluating dialogue systems through automatic metrics and human studies.
-   Highlights techniques to reduce the carbon footprint of model development and calls for sustainability practices.
-   Argues the importance of cross-disciplinary collaboration between researchers, policymakers, and civil society to guide the development of AI technologies.

## 💬 Other Comments

-   The safety techniques and red teaming process set a transparent and proactive risk mitigation standard. More details on the evolution of robustness during red teaming could further knowledge sharing.
-   The human evaluations provide a useful snapshot, but long-term studies are needed to characterize real-world impacts after deployment.
-   Discussing tensions between safety and capabilities highlights important open challenges in aligning models without reducing usefulness.
-   More analysis of the developed techniques on languages beyond English could better reveal multilingual issues and opportunities.

## 🙋‍♂️ Questions and Answers

**Q**: How was Llama 2 pretrained?  
**A**: Llama 2 was pretrained on 2 trillion tokens of publicly available data. The architecture uses an optimized transformer with improvements like a longer context length of 4k and grouped query attention for larger models. It was trained using AdamW optimization and checkpoint averaging.

**Q**: What techniques were used to create the helpful and safe Llama 2-Chat models?  
**A**: The authors used supervised fine-tuning (SFT) on high-quality human annotations to bootstrap. This was followed by iterative reinforcement learning with human feedback (RLHF) using preference learning and [PPO](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/ "Proximal Policy Optimization Algorithms (PPO and PPO2)") to align the models with human preferences on helpfulness and safety. Techniques like rejection sampling, targeted context distillation, and Ghost Attention were also used.

**Q**: How were the safety capabilities of Llama 2 measured?  
**A**: The safety of Llama 2 was measured through automatic benchmarks for truthfulness, toxicity, and bias, as well as human evaluations on adversarial prompts. Red teaming exercises were also conducted. The authors share a detailed analysis of aspects like pronoun usage, identity terms, and toxicity in the pretraining data.

**Q**: What were some key observations made during the development of Llama 2-Chat?  
**A**: Some observations include RLHF outperforming supervised tuning, the model learning to temporally organize knowledge and call APIs for tool usage with minimal examples, and the need to adjust temperature during iterative RLHF.

**Q**: What are some limitations of the human evaluations conducted?  
**A**: Limitations include the subjectivity of reviews, the scope of the prompt set, limiting generations to 1k tokens for some models, and assessing only final generations for conversations. Evaluations may also be biased towards the Llama 2-Chat models.

**Q**: How was the release strategy designed to be responsible?  
**A**: The authors performed extensive tuning focused on safety, shared detailed documentation to enable reproducibility and safety improvements, provided a responsible use guide, and released the models openly to enable innovation and reduce barriers.

**Q**: How were the reward models for RLHF trained and evaluated?  
**A**: Using human preference data, separate reward models were trained for safety and helpfulness. The models were evaluated on held-out sets and benchmark datasets. Techniques like margin loss and auxiliary safety loss improved accuracy.

**Q**: What mitigation strategies were used to improve model safety?  
**A**: Strategies included safety-focused SFT, targeted safety RLHF, context distillation with safety preprompts, rejection sampling to avoid unsafe responses, and using the safety reward model to choose whether to apply context distillation.

**Q**: What analysis was done on model bias during pretraining?  
**A**: The analysis looked at pronoun usage, identity terms, and toxicity in the pretrained data. The authors found some skews, like fewer she pronouns and Western-centric identities, that could lead to biased generations.

**Q**: How were human annotations collected and validated?  
**A**: Annotations were collected from trained annotators following detailed guidelines. A quality assurance process manually reviewed annotations before model training. Curriculum strategies were used to improve annotation quality.

**Q**: What limitations remain in the Llama 2 models?  
**A**: Limitations include lack of ongoing knowledge updates, potential for hallucinations, limitations in non-English languages, and risk of overly cautious responses due to safety tuning.

**Q**: What were some key results comparing Llama 2 to other models?  
**A**: Llama 2 outperformed open-source models like LLaMa, Falcon, and MPT on benchmarks. The 70B model was competitive with closed models like GPT-3 on some tasks. There are still significant gaps compared to very large models like GPT-4.

**Q**: How does the context length compare to other recent models? What was its impact?  
**A**: Llama 2 uses a 4k context length, compared to 2k in Llama 1 and often <2k in other models. This improved performance on long-context tasks like summarization while maintaining strong results on other tasks.

**Q**: What techniques were used to scale up training and inference?  
**A**: Checkpoint averaging, mixed precision, tensor parallelism, and optimizations like grouped query attention enabled scaling up training. Caching optimizations improved inference latency.

**Q**: How was the carbon footprint of pretraining calculated? What steps were taken to reduce it?  
**A**: Emissions were estimated based on GPU usage and emissions factors. Total emissions were 539 tCO2eq and fully offset. Compute and energy efficiency techniques reduced emissions, as did open release to avoid repeat training costs.

**Q**: What steps could be taken to improve Llama 2-Chat further?  
**A**: Additional tuning focused on non-English languages, handling tools/APIs, maintaining context over more turns, improving factual accuracy, and balancing safety with open-domain capabilities. More comprehensive evaluations post-deployment would also be beneficial.

**Q**: What safety risks were identified through red teaming exercises?  
**A**: Red teaming identified risks like providing dangerous advice when embedded in positive framing, distraction with creative prompts, responding dangerously despite initial reluctance, and non-English attack vectors.

**Q**: How were the human evaluations conducted? What were the limitations?  
**A**: Humans-rated model outputs from Llama 2-Chat and other models. Limitations included subjectivity, scope/diversity of the prompt set, and assessing only final turns for conversations.

**Q**: How does the contamination analysis account for the fragmentation of overlaps?  
**A**: The analysis looks for long, contiguous matches between evaluation and training data to avoid fragmentation. It tries multiple minimum match lengths and reports the longest that shows evidence of contamination.

**Q**: What techniques were used to improve multi-turn capabilities? How were they evaluated?  
**A**: Ghost Attention (GAtt) was proposed to help the model attend to key prompt attributes over multiple turns. GAtt improved consistency in human evaluations up to 20 turns.

**Q**: What steps were taken to enable reproducible release and research?  
**A**: The authors released full model details, training methodology, evaluation data, code examples, a responsible use guide, and a discussion of limitations to enable reproducibility and safe deployment.

**Q**: What was the motivation behind releasing Llama 2 openly?  
**A**: The authors aimed to encourage responsible AI innovation, draw on community wisdom for improving safety, increase access to foundational models, and consolidate costs so others don’t repeat expensive training.

**Q**: How does the helpfulness annotation process differ from existing datasets?  
**A**: High-quality annotation focusing on dialogue was prioritized over scale, with only tens of thousands of examples needed. Annotators compared model samples rather than provided demonstrations.

**Q**: What techniques mitigated forgetting during iterative RLHF?  
**A**: Using top samples from all prior iterations rather than just the latest helped avoid losing capabilities from previous versions.

**Q**: How did the authors balance safety and open-domain performance?  
**A**: By prefacing prompts when collecting unsafe demonstrations for SFT, safety could be taught without reducing capabilities. Targeted RLHF focused on safety also avoided regressing open-domain performance.

**Q**: What analyses were done to characterize the training data?  
**A**: Analyses examined language distributions, demographic representations based on pronouns and identity terms, and toxicity levels using classifier scores.
