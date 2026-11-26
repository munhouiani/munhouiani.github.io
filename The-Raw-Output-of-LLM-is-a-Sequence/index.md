---
title: "The Raw Output of LLM is a Sequence"
source: https://blog.munhou.com/The-Raw-Output-of-LLM-is-a-Sequence/
date: 2026-10-19
updated: 2026-11-25
tags: [Deep Learning, Machine Learning, Large Language Model]
---

# The Raw Output of LLM is a Sequence

I’m writing this post mostly as a reminder to myself. Even though I learned this concept a long time ago, I still find myself coming back to these fundamentals whenever a new training technique comes along.

## The Gap

![](https://blog.munhou.com/images/the-raw-output-of-llm-is-a-sequence/p1-sequence-in-sequence-out.svg)

Most tutorials illustrate next-token prediction like the picture on the left: text in, one token out. That’s fine as a mental model, but it doesn’t match what we see in the actual architecture and its outputs.

The actual input and output of an LLM are both sequences. Take the input “I love”. To keep things simple, one word is one token. The model returns two rows of logits:

-   **Row 1** sees only “I” and predicts what comes after “I”.
-   **Row 2** sees “I love” and predicts what comes after “I love”.

The causal mask makes this work. Position $t$ can attend only to positions $\\le t$, so every row makes a prediction without looking at the answer.

Also note what the raw output is: **logits**, not tokens. Softmax turns each row into a probability distribution. Choosing a token, by argmax or by sampling, is a separate step that happens outside the model.

```python
logits = model(ids)          # ids: [1, 2]  ->  logits: [1, 2, V]
probs = logits.softmax(-1)   # one distribution per position
```

Let’s demonstrate with actual code.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# load tokenizer and model from huggingface
model_name = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# the input text
input_text = "The capital of France is Paris, and the capital of Japan is"
input_ids = tokenizer(text=input_text, return_tensors="pt").input_ids
token_to_string = [tokenizer.decode(token_id) for token_id in input_ids[0]]

print(f"The input: {input_text}")
print(f"input tokens: {input_ids}")
print(f"input tokens shape: {input_ids.shape}")
print("string to token mapping")
for t, (i, s) in enumerate(zip(input_ids[0].tolist(), token_to_string)):
    print(f"    [{t:>2}] {s:<16} {f'id={i}'}")
print()

# the output
logits = model(input_ids=input_ids).logits.float()
print("The output (one forward pass):")
print(f"input tokens shape: {input_ids.shape}")
print(
    f"output logits shape: {logits.shape} <- [batch, T={input_ids.shape[1]}, vocab V={logits.shape[-1]}]"
)
print(
    f"{input_ids.shape[1]} tokens in, {logits.shape[1]} tokens out. Each row is a score for every one of the {tokenizer.vocab_size} vocab entries: the model's guess for the token that comes AFTER that position."
)
print()

# decode every row
probs = logits[0].softmax(-1)
p_top, pred = probs.max(-1)
print("Every row = next-token prediction for its prefix")
head = f"  {'t':>3}  {'model sees input[0..t]':<{len(input_text)}}  {'predicts':<14} {'p':>5}   actual input[t+1]"
hits = 0
for t in range(input_ids.shape[1]):
    seen = tokenizer.decode(input_ids[0, : t + 1])
    guess = tokenizer.decode(pred[t])
    line = (
        f"  {t:>3}  {seen:<{len(input_text)}}  {guess:<14} {p_top[t].item():>5.2f}   "
    )
    if t < input_ids.shape[1] - 1:
        ok = pred[t].item() == input_ids[0, t + 1].item()
        hits += ok
        line += ("✓ " if ok else "✗ ") + token_to_string[t + 1]
    if t == input_ids.shape[1] - 1:
        line = line + "   <- the 'one next word'"
    print(line)
print(
    f"\n  {hits}/{input_ids.shape[1] - 1} rows guessed the token that really comes next in the input."
)
print(
    "The 'one next word' everybody expects is only the LAST row. Training grades ALL rows at once (labels = input shifted left by one)."
)
```

```plaintext
The input: The capital of France is Paris, and the capital of Japan is
input tokens: tensor([[1784, 8961, 1307, 5498, 1395, 6993, 1044, 1321, 1278, 8961, 1307, 6777,
         1395]])
input tokens shape: torch.Size([1, 13])
string to token mapping
    [ 0] The              id=1784
    [ 1]  capital         id=8961
    [ 2]  of              id=1307
    [ 3]  France          id=5498
    [ 4]  is              id=1395
    [ 5]  Paris           id=6993
    [ 6] ,                id=1044
    [ 7]  and             id=1321
    [ 8]  the             id=1278
    [ 9]  capital         id=8961
    [10]  of              id=1307
    [11]  Japan           id=6777
    [12]  is              id=1395

The output (one forward pass):
input tokens shape: torch.Size([1, 13])
output logits shape: torch.Size([1, 13, 131072]) <- [batch, T=13, vocab V=131072]
13 tokens in, 13 tokens out. Each row is a score for every one of the 131072 vocab entries: the model's guess for the token that comes AFTER that position.

Every row = next-token prediction for its prefix
    0  The                                                          ,               0.23   ✗  capital
    1  The capital                                                   of             0.35   ✓  of
    2  The capital of                                                the            0.20   ✗  France
    3  The capital of France                                         is             0.98   ✓  is
    4  The capital of France is                                      Paris          0.86   ✓  Paris
    5  The capital of France is Paris                               ."              0.54   ✗ ,
    6  The capital of France is Paris,                               which          0.26   ✗  and
    7  The capital of France is Paris, and                           the            0.49   ✓  the
    8  The capital of France is Paris, and the                       capital        0.33   ✓  capital
    9  The capital of France is Paris, and the capital               of             0.99   ✓  of
   10  The capital of France is Paris, and the capital of            Germany        0.17   ✗  Japan
   11  The capital of France is Paris, and the capital of Japan      is             1.00   ✓  is
   12  The capital of France is Paris, and the capital of Japan is   Tokyo          0.99      <- the 'one next word'

  7/12 rows guessed the token that really comes next in the input.
The 'one next word' everybody expects is only the LAST row. Training grades ALL rows at once (labels = input shifted left by one).
```

## Training Uses Every Row

During training the targets are simply the input shifted by one position. Each row is scored against the token that really comes next in the data:

| Position | What it sees | Its target |
| --- | --- | --- |
| 1 | I | love |
| 2 | I love | the next word in the data, e.g. “cats” |

So one forward pass over $T$ tokens gives $T$ training signals, all computed in parallel. Every position is a training example. This is why transformers train so efficiently.

We can see this actual code at HuggingFace’s [`transformers`](https://github.com/huggingface/transformers/blob/eeed120a20c46df4bda9d955014d29114f521849/src/transformers/loss/loss_utils.py#L32-L71) library:

```python
# ForCausalLMLoss at src/transformers/loss/loss_utils.py

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    ...
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
```

On an older version of `transfomers` library, before the loss was moved into a shared function, the lines are even more readable. The following snippet is taken from `modeling_llama.py` of version [`4.44.2`](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L1213-L1222):

```python
# Shift so that tokens < n predict n
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss_fct = CrossEntropyLoss()
...
loss = loss_fct(shift_logits, shift_labels)
```

## Everything Else is a Variation

Every other stage, be it SFT, knowlege distillation or RL, does the same thing: feed a sequence in, get one distribution per position back, apply cross entropy. The only change is which positions count, what the target is, and how much each position weighs.

| Stage | Positions that count | Target at each position | Weight |
| --- | --- | --- | --- |
| Pretraining, mid-training | All | Next token in the data | 1 |
| SFT | Assistant tokens | Next token in the data | 1 |
| Knowledge distillation | Assistant tokens | The teacher’s full distribution | 1 |
| RL (GRPO) | Tokens the model generated | The token the model sampled | The rollout’s advantage $A$ |

If you think this post is helpful please share on the comment section.

Thanks.
