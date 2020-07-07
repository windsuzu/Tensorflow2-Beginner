# Sentiment in text

我們將文字的情感分類，作為第一個實作在文字上的任務，但首先要怎麼將文字表達成一堆數字呢 ?

## Word based encodings

第一個想到把文字用數字表達的方法是 ASCII code

![](../../assets/text_representation_ascii.png)

但這個方法沒有辦法表達出文字的意義，甚至上面兩個字的 ASCII 看起來很像但在意義上卻有蠻大的差別

![](../../assets/text_representation_dict.png)

第二個方法我們把文字標上號碼，只看號碼的時候似乎就能看出數字的相似度 (和句子關聯度成正比)

### API

Tensorflow 提供 API 能夠實作上面的第二個方法，快速標註每個單字的號碼

``` python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['i love my dog',
             'I, love my cat',
             'You love my dog!' ]


tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
tokenizer.word_index

# {'cat': 5, 'dog': 4, 'i': 3, 'love': 1, 'my': 2, 'you': 6}
```

1. 建立 Tokenizer 要輸入 `num_words=n` 參數，建立一個大小為 n 的字典來存放 **token**
   1. 單字數超過 n 時，會自動選前 n 最常出現的單字
2. Tokenizer 的 `fit_on_texts()` 會將餵入句子的每個單字轉換成 **token**
   1. 會自動小寫
   2. 會自動去掉標點符號
3. `word_index` 這個 property 可以顯示字典的單字和 **token** 對應表

## Text to sequence

Tokenizer 知道各種單字的號碼後，就可以把各種句子轉成號碼表示

做到這件事情的是 `tokenizer.texts_to_sequences(sequences)`

``` python
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# [[3, 1, 2, 4], [3, 1, 2, 5], [6, 1, 2, 4]]
```

其他句子一樣通用，但不認識的單字會被略過

``` python
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_data = tokenizer.texts_to_sequences(test_data)
print(test_data)

# [[3, 1, 2, 4], [2, 4, 2]]
```

所以若 tokenizer 學得不夠多，句子就會完全失去意義，例如第二個的 `'my dog loves my manatee'` 變成了 `'my dog my'`

### OOV_TOKEN

一個做法是在建立 `Tokenizer` 的時候加入 `oov_token`，這樣以後沒在字典的字就會變成該 token

``` python
tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
# {'<OOV>': 1, 'love': 2, 'my': 3, 'i': 4, 'dog': 5, 'cat': 6, 'you': 7}

sequences = tokenizer.texts_to_sequences(test_data)
# [[4, 1, 2, 3, 5], [3, 5, 1, 3, 1]]
```

可以看到 `'my dog loves my manatee'` 變成了 `'my dog <OOV> my <OOV>'`

但這不是最佳解，我們應該要想辦法讓字典學到更多字，避免這種情形

## Padding

在訓練前還有最後一項工作，就是讓每個不同長度的句子變得一樣長，像圖片的 `resize` 一樣

做法是使用 Tensorflow 裡面 `tensorflow.keras.preprocessing.sequence` 中的 `pad_sequences`

``` python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# {'<OOV>': 1,  'amazing': 11,  'cat': 7,  'do': 8,  'dog': 4,  'i': 5,
# 'is': 10,  'love': 3,  'my': 2,  'think': 9,  'you': 6}


sentences = tokenizer.texts_to_sequences(sentences)
# [[5, 3, 2, 4], 
#  [5, 3, 2, 7], 
#  [6, 3, 2, 4], 
#  [8, 6, 9, 2, 4, 10, 11]]


padded = pad_sequences(sentences)
# [[ 0  0  0  5  3  2  4]
#  [ 0  0  0  5  3  2  7]
#  [ 0  0  0  6  3  2  4]
#  [ 8  6  9  2  4 10 11]]


padded = pad_sequences(sentences, padding='post', maxlen=5, truncating='post')
# [[5 3 2 4 0]
#  [5 3 2 7 0]
#  [6 3 2 4 0]
#  [8 6 9 2 4]]
```

pad_sequences 會把多餘的地方填滿 0，裡面有幾個參數可以給定

1. padding 可以決定是從前面還是在後面填 0
   1. pre (default)
   2. post
2. maxlen 可以決定所有句子最長的長度
   1. None (default, 預設是抓最長的那個句子長度)
   2. int
3. truncating 可以決定當句子大於 maxlen 時，要從前面還是後面切掉
   1. pre (default)
   2. post

## Real Word Dataset - Sarcasm

這邊使用一個 [kaggle](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home) 上的任務，要分類句子是否在諷刺

1. 先來載入 json 的資料集
2. 建立 `Tokenizer`
3. 用 `Tokenizer` 和 `sentences` 來建立字典 `word_index`
4. 將 `sentences` 都轉成 token 格式
5. padding 句子

### Data Loading

``` python
!wget --no-check-certificate https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json -O /tmp/sarcasm.json

import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# sentences
# ["former versace store clerk sues over secret 'black code' for minority shoppers",
#  "the 'roseanne' revival catches up to our thorny political mood, for better and worse",
#  "mom starting to fear son's web series closest thing she will have to grandchild",
#  ... ]

# labels (1: sarcastic, 0: otherwise)
# [0, 0, 1, ...]
```

### Tokenize

``` python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
# 29657

print(word_index)
# {'<OOV>': 1, 'to': 2, 'of': 3, ..., 'gourmet': 29656, 'foodie': 29657}

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded.shape)
# (26709, 40)

print(setences[0])
print(padded[0])
# former versace store clerk sues over secret 'black code' for minority shoppers
# [  308 15115   679  3337  2298    48   382  2576 15116     6  2577  8434
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0     0     0     0     0     0     0     0     0
#      0     0     0     0]
```

# Exercise

在 exercise 1 要做的是對 [BBC text archive](http://mlg.ucd.ie/datasets/bbc.html) 資料做預處理

1. tokenize the dataset
2. remove common stopwords
   1. great source of these stop words can be found [here](https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js)

[Exercise 1 的解答在這裡](exercise1.ipynb)