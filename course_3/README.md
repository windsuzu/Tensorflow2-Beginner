# Natural Language Processing in TensorFlow

## Week 1: Sentiment in text

在訓練文字資料前，要把文字都變成數字 (tokenization)，並且做些預處理，例如建立字典、對句子做 padding 等

[Note is here](week1)

## Week 2: Word Embeddings

有個 tokenization 方式叫做 embedding，能夠使用高維度向量來更加表示文字之間的關係 

另外來試著預測看看 IMDb 的評論是好評還是負評

[Note is here](week2)

## Week 3: Sequence models

一般的神經網路模型來訓練文字預測通常會喪失整個句子的意義，這時就要改用 RNN 這類的 Sequence models 來實現句子意義的傳遞

[Note is here](week3)

## Week 4: Sequence models and literature

我們綜合前三周所學，讓模型不斷預測句子的下一個字，然後產生出一首完整的詩歌或文章

[Note is here](week4)