# RambaNet

We aim to perform authorship attribution on medieval Jewish Thought books using modern Deep Learning approach. This is a classification problem, when the challenge is the modeling of authorship characteristics.

The use of Neural Networks and semantically-inclined embeddings allows to take into account not only stylography but also content and ideas.

A side analysis is the analysis of the relevancy of the Deep Learning models to non-Roman or Anglo-Saxon languages. Medieval Hebrew possesses properties which may challenge embedding methods like GloVe and word2vec. For example, there may be several different orthographs for the same word; prepositions, linking words, determiners and possessive adjectives are often part of other words and not independent words; the lack of ponctuation.

## Background

### Classifier

There are comparatively few papers researching neural networks for the purpose of authorship attribution. The existing literature (a list can be found in the References below) can be divided into two main approachs:

* CNNs.
* LSTMs and GRUs, the latter giving consistently better results.
<!--
### Embeddings

## Dataset




-->
## References
1. Sebastian Ruder, Parsa Ghaffari, John G. Breslin, "Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution", [_arXiv:1609.06686_](https://arxiv.org/abs/1609.06686) (2016).
1. Chen Qian, Tianchang He, Rao Zhang, ["Deep Learning based Authorship Identification"](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf), Stanford University (2017).
1. Asad Mahmood, ["Authorship Attribution using CNNs"](https://github.com/asad1996172/Authorship-attribution-using-CNN), GitHub.
1. Aisha Khatun, Anisur Rahman, Md. Saiful Islam, Marium-E-Jannat, "Authorship Attribution in Bangla literature using Character-level CNN",[ _arXiv:2001.05316_](https://arxiv.org/abs/2001.05316) (2020)