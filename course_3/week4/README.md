# Sequence models and literature

## Introduction

RNN 當中一個有趣的應用是讀過某個類型的文章，就可以依照該文章的風格產生出相似的作品 (e.g., shakespeare)

產生文章聽起來很難，其實跟前面幾週一樣只是在進行預測的動作而已

``` python
x = 'Twinkle Twinkle little'
y = 'star'
```

例如告訴模型我們輸入的 `Twinkle Twinkle little` 要對應的答案是 star

丟入大量類似的 corpus 進入模型訓練，未來模型就會模仿他學習過的東西創造出句子

## Preprocessing

首先要 tokenize 整個文章，單字數 +1 表示考慮進 OOV 的字，總共 263 個單字

``` python
data = "In the town of Athy one Jeremy Lanigan \n Battered away til ..."

tokenizer = Tokenizer()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
# {'and': 1, 'the': 2, 'a': 3, 'in': 4, ... }

print(total_words)
# 263
```

在訓練時，會把句子拆成好幾個 xs 和 ys 來訓練，每次採用最後一個單字當作 y

```
original text: 
In the town of Athy one Jeremy Lanigan


(2) In the
x: in
y: the

(3) In the town
x: in the
y: town

(4) In the town of
x: in the town
y: of

...

(8) In the town of Athy one Jeremy Lanigan
x: in the town of athy one jeremy
y: lanigan
```

所以 8 個單字的句子，可以拆出 7 個 training dataset，每個 dataset 再做 padding 來方便拆分 x, y 

``` python
[4, 2, 66, 8, 67, 68, 69, 70] =>

[0, 0,  0, 0,  0,  0,  4,  2],
[0, 0,  0, 0,  0,  4,  2, 66],
[0, 0,  0, 0,  4,  2, 66,  8],
[0, 0,  0, 4,  2, 66,  8, 67],
[0, 0,  4, 2, 66,  8, 67, 68],
[0, 4,  2, 66, 8, 67, 68, 69],
[4, 2, 66, 8, 67, 68, 69, 70] =>

x                           y
[0, 0,  0, 0,  0,  0,  4],  [ 2]
[0, 0,  0, 0,  0,  4,  2],  [66]
[0, 0,  0, 0,  4,  2, 66],  [ 8]
[0, 0,  0, 4,  2, 66,  8],  [67]
[0, 0,  4, 2, 66,  8, 67],  [68]
[0, 4,  2, 66, 8, 67, 68],  [69]
[4, 2, 66, 8, 67, 68, 69],  [70]
```

程式碼分成四個步驟: 

1. extract n-grams

``` python
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

print(input_sequences) 
# [[4, 2], [4, 2, 66], [4, 2, 66, 8], [4, 2, 66, 8, 67], ... ]
```

2. padding n-grams

``` python
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

print(input_sequences)
# [[  0   0   0 ...   0   4   2]
#  [  0   0   0 ...   4   2  66]
#  ...
#  [  0   0   0 ... 262  13   9]
#  [  0   0   0 ...  13   9  10]]
```

3. seperate (x, y) pairs

``` python
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

print(xs)
# array([[  0,   0,   0, ...,   0,   0,   4],
#        [  0,   0,   0, ...,   0,   4,   2],
#        ...,
#        [  0,   0,   0, ...,  60, 262,  13],
#        [  0,   0,   0, ..., 262,  13,   9]], dtype=int32)

print(labels)
# array([  2,  66, ...,  9,  10], dtype=int32)
```

4. use `tf.keras.utils.to_categorical` to one-hot labels

``` python
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

print(ys[0])
# [0. 0. 1. 0. 0. ... 0. 0.] 
# only index 2 is 1 out of 263 elements

print(ys.shape)
# (453, 263)
```

## Training

https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%201%20-%20Notebook.ipynb


## Text Generation (Prediction)



# generating text using a Character-based RNN

https://www.tensorflow.org/tutorials/text/text_generation