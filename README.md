# RambaNet

We aim to perform authorship attribution on medieval Jewish Thought books using modern Deep Learning approach. This is a classification problem, when the challenge is the modeling of authorship characteristics.

The use of Neural Networks and semantically-inclined embeddings may allow to take into account not only stylography but also content and ideas.

## Background and Methodoloy

### Classifier

There are comparatively few papers researching neural networks for the purpose of authorship attribution. The existing literature (a list can be found in the References below) can be divided into two main approachs:

* Convolutional Neural Networks (CNN).
* Recurrent Neural Networks (RNN) such as LSTM and GRU, the latter giving consistently slightly better results.

An extensive litterature review (see reference) seems to lead to the conclusion that CNNs are better suited for the task. The explanation may be that RNNs are great at learning temporal connections between words, which is appropriate for predicting the next word in a sentence. Unlike CNNs, they are not designed to capture semantic or stylistic information.

### Embeddings

Generally in NLP embeddings are created for words or sentences. However, the latest studies [TODO] seem to suggest that using character-level embeddings may lead to better results. Moreover, word embeddings were crafted with Enligh (and other Roman or Anglo-Saxon languages) in mind, where prepositions, linking words, determiners and possessive adjectives are always separate words and have low semantic meaning.

Medieval Hebrew possesses properties which may challenge word embedding methods like GloVe and word2vec. For example, there may be several different orthographs for the same word; prepositions, linking words, determiners and possessive adjectives are often part of other words and not independent words. 

Recent research in Bengali [Khatun 2020] suggests that character-level embeddings lead to better accuracy. Bengali shares some characteristics with Hebrew.

Moreover, homonyms and homographs are frequent in Hebrew, much more than in English (see [this paper](https://m.tau.ac.il/~pelegor/pdfs/15.%20Peleg,%20Edelist,%20Eviatar,%20&%20Bergerbest,%20in%20press.pdf), pp. 3-4 and [that paper](https://www.academia.edu/35145231/The_Vocabulary_of_Classical_Hebrew_New_Facts_and_Figures) pp. 9-10); and the number of byforms (alternative spellings of words) is astoninishgly high (more than 30% of Classical Hebrew according to [that paper](https://www.academia.edu/35145231/The_Vocabulary_of_Classical_Hebrew_New_Facts_and_Figures)).

Lastly, an important drawback of word embeddings is that the vocabulary must be defined in advance, and less common words are left out of the embedding. However, for authorship attribution purposes the study of hapax legomena embeds a lot of information.


## Dataset

We use the JSON Hebrew version of the dataset [Sefaria-Export](https://github.com/Sefaria/Sefaria-Export), which is graciously provided by Sefaria. The raw dataset is not included in the repository due to its size.


TODO: preprocessing

## Results

### Character-Level CNN



## References
1. Sebastian Ruder, Parsa Ghaffari, John G. Breslin, "Character-level and Multi-channel Convolutional Neural Networks for Large-scale Authorship Attribution", [_arXiv:1609.06686_](https://arxiv.org/abs/1609.06686) (2016).
1. Chen Qian, Tianchang He, Rao Zhang, ["Deep Learning based Authorship Identification"](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf), Stanford University (2017).
<!--1. Asad Mahmood, ["Authorship Attribution using CNNs"](https://github.com/asad1996172/Authorship-attribution-using-CNN), GitHub.-->
1. Xiang Zhang, Junbo Zhao, Yann LeCun, "Character-level Convolutional Networks for Text Classification", [_ 	arXiv:1509.01626_](https://arxiv.org/abs/1509.01626v3) (2015).
1. Aisha Khatun, Anisur Rahman, Md. Saiful Islam, Marium-E-Jannat, "Authorship Attribution in Bangla literature using Character-level CNN",[ _arXiv:2001.05316_](https://arxiv.org/abs/2001.05316) (2020).
