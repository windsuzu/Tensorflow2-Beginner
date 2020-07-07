# Word Embeddings

除了用號碼表達每個文字，還可以使用 word embedding 來表達每個文字，能夠更準確的抓出文字之間的關聯性

## Introduction

我們可以用 convolution 來擷取圖片的特徵，在文字任務中也有相似功用的工具: Embeddings

Embeddings 可以將文字表示為高維度的向量，在這向量空間中，類似涵義的文字的向量就會靠近在一起

我們可以在 [projector.tensorflow.org](http://projector.tensorflow.org/) 查看訓練出來的 embedding 空間

![](../../assets/embedding_projector.png)


===



imdb dataset

tokenizer

classifier

showing up embedding and visualize

===

do it again in sarcasm

tuning

===

pre tokenized dataset

subword

not work but in RNN

