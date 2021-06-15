# BERT

Collection of scripts and notes regarding the _Bidirectional Encoder Representation of Transformers_ (or short BERT) Architecture for self-educational purposes.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Fundamentals & Notes](#fundamentals--notes)
- [Get Started](#get-started)
- [Links](#links)
  - [Scientific Papers](#scientific-papers)
  - [GitHub Repositories](#github-repositories)
  - [Misc](#misc)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Fundamentals & Notes

- **Transformer Architecture** by Vaswani et. Al.
  - 6 Encoder + 6 Decoder <-> BERT: 12 x Encoder
  - 8 attention heads (MHA) randomly initialized with length 64 --> 512d embeddings
  - designed with the goal of machine translation in mind
  - incorporates positional encoding for attention with hardcoded hyperbolic functions
- **BERT** by Devlin et. Al.
  - possibility for transfer learning and benefit from pre-trained model
  - _Transfer Learning_: fast prototyping, requires less data, outperforming classical methods
  - made out of transformers without recurrence (RNNs) and has its own tokenizer as well as a fixed vocabulary with 30k tokens
  - common misspellings are not included in the vocab since it it is pre-trained on Wikipedia & Book Corpus
  - breaks down unknown words into subword tokens; subword tokens start with two hashtags except the first subtoken
  - contains 109M parameters (417MB on disk) --> slow fine-tuning/inferencing
  - _distillation_ (removing of parameters while trying to keep accuracy) is workaround for slow inference
  - trained on _Masked Language_ (MLM) and _Next Sentence Prediction_ (NSP) tasks
    - MLM and NSP is carried out in parallel and hence includes masked, random tokens
    - masking 12% of the training set tokens
    - exchanging 1.5% of the tokens with random tokens (unigram-sampled: more likely words have a higher chance to replace a token) --> prevents overfitting?
    - flagging 1.5% of the tokens for prediction
  - self attention: _"looking at other words in the input sentence while encoding a specific word"_
    - train query, key and value weight matrices W_q, W_k and W_v
    - calculate three vectors for each of the input vectors --> Query, Key, Value
    - calculate score by taking dot product of query vector q and key vector k of every input vector of the sentence
    - divide score by sqrt of d_k in order to obtain stable gradients & compute softmax
    - reminder: attention is O(n^2) in complexity with sequences of length n --> long sentences are very expensive
  - _Multi-headed Attention_
    - expands model's focus abilities on different positions
    - provides multiple 'representation subspaces'
  - 12 attention heads with length 64 --> 768d embeddings: one BERT token (word embedding) has 768 features
  - BERT embedding consists out of the sum of three different _embedding types_:
    - _Positional Encoding_ to incorporate the relative position as a feature for attention;
      these positional embeddings are learned and not hardcoded with hyperbolic functions as in the transformer architecture
    - _Segment Embedding_ which differentiates the sentences/segments from each other
    - _Vocab Embedding_ which correlates to the embedding of a given token (--> sub/word)
  - _Layer Types_:
    1. Self-Attention Layers
    2. Attention Output Layers (768 Neurons)
    3. Indermediate / Dense Layers (3072 Neurons / 2.4M Weights)
    4. Output Layer (768 Neurons / 2.4M Weights)
  - applicable: classification, NER, POS-tagging or QnA
  - not applicable: language model, text generation, machine translation

## Get Started

1. Create a virtual environment with the essential libraries that are defined in `requirements.txt`
2. Download data sets with the `scripts/dataloader.py`
3. Execute BERT scripts (preferrably on a GPU)

## Links

### Scientific Papers

- [BERT](https://arxiv.org/pdf/1810.04805.pdf) by Devlin et. Al.
- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) by Vaswani et. Al.

### GitHub Repositories

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BERT](https://github.com/google-research/bert) by Google

### Misc

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) by Rush et. Al.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Allamar
- Chris McCormick's excellent [content](https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw/featured)
