---
title: "Byte-Pair-Encoding (BPE)"
source: https://blog.munhou.com/Byte-Pair-Encoding-BPE/
date: 2023-08-14
updated: 2026-11-24
---

# Byte-Pair Encoding

## What is Byte-Pair Encoding

Byte-pair encoding (BPE) is a compression algorithm that can be used for text tokenization.

## How It Works

-   The algorithm starts by splitting the text into individual characters.

```text
aaabdaaabac
```

-   It then iteratively merges the most frequent pair of tokens (initially characters) into a new token.

```text
ZabdZabac
Z=aa
```

-   This merging process continues until a target vocabulary size is reached.
-   The most frequent tokens get merged first, creating common subwords.
-   Less frequent words get split into multiple tokens.
-   When applying BPE, the text is encoded as a sequence of tokens from the final BPE vocabulary.

## Key Properties

-   It balances vocabulary size while capturing the most common subword units.
-   Morphologically related words get consistent tokenized representations.
-   Rare and unknown words are automatically split into meaningful subwords.
-   The final vocabulary size can be controlled.
-   BPE creates an open vocabulary that can handle new words.
-   Simple and efficient to learn and apply.

## Differences Between The Original Algorithm And Modification

There are a few key differences between the original byte-pair encoding (BPE) algorithm and how it is typically used for tokenization in large language models:

-   Open vs. closed vocabulary: The original BPE produces a closed vocabulary based on the training corpus. For language models, an open vocabulary is needed to handle unseen words. So, BPE is modified to allow encoding unknown characters as individual tokens.
-   Iterative merging: Original BPE does iterative merging of symbols until it reaches the target vocabulary size. The number of merge operations for language models is fixed based on heuristics rather than a size target.
-   Unicode handling: The original BPE works at the byte level. For text tokenization, BPE is modified to work at the Unicode character level while handling variable-length encodings.
-   Digit handling: Digits are often separated into individual tokens in text BPE rather than merged. This provides consistency across numbers.
-   Casing: Language model BPE tokenizers often preserve the original word casing rather than lowercasing.
-   Granularity: Language model BPE tokenizers use larger vocabularies (10k-100k tokens) than the original algorithm. This provides greater coverage of subword units.

### Support for Open Vocabulary

-   Escape character token: The BPE vocabulary can include a special “escape character” token like `<unk>` to represent any unknown character. Encoding replaces any character not in the base vocabulary with this escape token.
-   Character fallback: When an unknown character is encountered during encoding, it is converted into a single-character token. This allows handling any Unicode characters not seen in training.
-   Hybrid approach: A hybrid of the above two approaches can be used. First, unknown characters are converted to single-character tokens. But if the frequency of an unknown character exceeds a threshold, it is added to the vocabulary, and the existing tokens are updated.

### Modified Merge Process

-   The number of merge operations is fixed ahead of time, based on heuristics like 30k or 50k merges. It is not determined by a target vocabulary size.
-   Certain symbols, like digits, are prevented from merging. This keeps them as consistent individual tokens.
-   Casing information can be retained by preventing case variants from merging.
-   Uncommon characters are often excluded from merging to limit vocabulary growth.
-   The final vocabulary size is much larger, ranging from 10k to 100k tokens.

### Unicode Handling

-   Unicode symbols are treated as atomic units rather than individual bytes. So, multi-byte encoded characters are kept together.
-   Vocabulary symbols correspond to Unicode code points rather than byte values.
-   Merge operations are based on Unicode characters regardless of variable encoding lengths.
-   Characters outside the Basic Multilingual Plane (BMP) are supported seamlessly.
-   Unknown or uncommon characters default to single-character symbols rather than escapes.
-   Casing, diacritics, and punctuation are mostly preserved rather than discarded.
-   Whitespace tokens delimit words rather than space characters.
-   The model learns contextual representations for rare symbols.
-   Special handling for common non-alphabet characters like digits, quotes, etc.

## References

1.  [https://en.wikipedia.org/wiki/Byte\_pair\_encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
2.  [https://en.wikipedia.org/wiki/Plane\_(Unicode)](https://en.wikipedia.org/wiki/Plane_\(Unicode\))
