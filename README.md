# RambaNet

The goal of the project is to perform authorship attribution on medieval Jewish Thought books using a modern Deep Learning approach. This is a classification problem, when the challenge is the modeling of authorship characteristics.

The use of Neural Networks and semantically-inclined embeddings may allow to take into account not only stylography but also content and ideas.

## Dataset

We use the JSON Hebrew version of the dataset [Sefaria-Export](https://github.com/Sefaria/Sefaria-Export), which is graciously provided by Sefaria. The raw dataset is not included in the repository due to its size.

**Some basic data analysis and vizualizations of the dataset may be found in the folder `basic_data_analysis/`.**

## Background and Methodoloy

### Classifier

There are comparatively few papers researching neural networks for the purpose of authorship attribution. The existing literature (a list can be found in the References below) can be divided into two main approachs:

* Convolutional Neural Networks (CNN).
* Recurrent Neural Networks (RNN) such as LSTM and GRU, the latter giving consistently slightly better results.

An extensive litterature review (see reference) seems to lead to the conclusion that CNNs are better suited for the task. The explanation may be that RNNs are great at learning temporal connections between words, which is appropriate for predicting the next word in a sentence. Unlike CNNs, they are not designed to capture semantic or stylistic information.

### Embeddings

Generally in NLP embeddings are created for words or sentences. However, recent studies [Zhang 2015] seem to suggest that character-level embeddings provide competetitive results while being less complex, requiring less parameters and training time. In this model we choose to research character embeddings, for two main reasons.

Firstly, an important drawback of word embeddings is that the vocabulary must be defined in advance, and out-of-vocabulary and less common words are left out of the embedding. However, for authorship attribution purposes the study of hapax legomena (unique words) or just uncommon expressions bears a lot of information.

Second, the method of embedding words was crafted with Engligh (or other Roman or Anglo-Saxon languages) in mind. In English,prepositions, linking words, determiners and possessive adjectives are always separate words and have low semantic meaning. Medieval Hebrew possesses properties which may challenge word embedding methods like GloVe and word2vec. Prepositions, linking words, determiners and possessive adjectives are often part of other words and not independent words. Moreover, homonyms and homographs are frequent in Hebrew, much more than in English (see [this paper](https://m.tau.ac.il/~pelegor/pdfs/15.%20Peleg,%20Edelist,%20Eviatar,%20&%20Bergerbest,%20in%20press.pdf), pp. 3-4 and [that paper](https://www.academia.edu/35145231/The_Vocabulary_of_Classical_Hebrew_New_Facts_and_Figures) pp. 9-10). The number of byforms (alternative spellings of words) is astoninishgly high (more than 30% of Biblical Hebrew according to [that paper](https://www.academia.edu/35145231/The_Vocabulary_of_Classical_Hebrew_New_Facts_and_Figures)).

Recent research in Bengali [Khatun 2020] suggests that character-level embeddings lead to better accuracy. Bengali shares some characteristics with Hebrew [SOURCE?].

## Results

### Character-Level CNN

## Requirements

- Python 3.7
- PyTables
- TensorFlow


## References
1. Sebastian Ruder, Parsa Ghaffari, John G. Breslin, "Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution", [_arXiv:1609.06686_](https://arxiv.org/abs/1609.06686) (2016).
1. Chen Qian, Tianchang He, Rao Zhang, ["Deep Learning based Authorship Identification"](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf), Stanford University (2017).
1. Xiang Zhang, Junbo Zhao, Yann LeCun, "Character-level Convolutional Networks for Text Classification", [_arXiv:1509.01626_](https://arxiv.org/abs/1509.01626v3) (2015).
1. Aisha Khatun, Anisur Rahman, Md. Saiful Islam, Marium-E-Jannat, "Authorship Attribution in Bangla literature using Character-level CNN",[ _arXiv:2001.05316_](https://arxiv.org/abs/2001.05316) (2020).
<!--1. Asad Mahmood, ["Authorship Attribution using CNNs"](https://github.com/asad1996172/Authorship-attribution-using-CNN), GitHub.-->