# BERT

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Fundamentals & Notes](#fundamentals--notes)
- [Links](#links)
  - [Scientific Papers](#scientific-papers)
  - [GitHub Repositories](#github-repositories)
  - [Misc](#misc)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Fundamentals & Notes

- **BERT**
  - possibility for transfer learning and benefit from pre-trained model
  - _Transfer Learning_ fast prototyping, requires less data, outperforming classical methods
  - made out of transformers without recurrence (RNNs) and has its own tokenizer as well as a fixed vocabulary with 30k tokens
  - common misspellings are not included in the vocab since it it is pre-trained on Wikipedia & Book Corpus
  - breaks down unknown words into subword tokens; subword tokens start with two hashtags except the first subtoken
  - one BERT token (word embedding) has 768 features
  - contains 109M parameters (417MB on disk) --> slow fine-tuning/inferencing
  - _distillation_ (removing of parameters while trying to keep accuracy) is workaround for slow inference
  - trained on _Masked Language_ and _Next Sentence Prediction_ tasks
  - self attention: _"looking at other words in the input sentence while encoding a specific word"_
  - applicable: classification, NER, POS-tagging or QnA
  - not applicable: language model, text generation, machine translation
- **Transformer**
  - 6 Encoder + 6 Decoder <-> BERT: 12 x Encoder
  - designed with the goal of machine translation in mind

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
