我想記錄我在 coursera 學習由 deeplearning.ai 提供的 Tensorflow 課程記錄 !

影片網址在 https://www.coursera.org/specializations/tensorflow-in-practice

由 Google 的 Laurence Moroney 講課，一共分成四大部分:

* Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
* Convolutional Neural Networks in TensorFlow
* Natural Language Processing in TensorFlow
* Sequences, Time Series and Prediction

# Catalog

* [Course 1: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](course_1)
  * [Week 1: A New Programming Paradigm](course_1/week1)
  * [Week 2: Introduction to Computer Vision](course_1/week2)
  * [Week 3: Enhancing Vision with Convolutional Neural Networks](course_1/week3)
  * [Week 4: Using Real-World Images](course_1/week4)
* [Course 2: Convolutional Neural Networks in TensorFlow](course_2)
  * [Week 1: Exploring a Larger Dataset](course_2/week1)
  * [Week 2: Augmentation: A technique to avoid overfitting](course_2/week2)
  * [Week 3: Transfer Learning](course_2/week3)
  * [Week 4: Multiclass Classifications](course_2/week4)
* [Course 3: Natural Language Processing in TensorFlow](course_3)
  * [Week 1: Sentiment in text](course_3/week1)
  * [Week 2: Word Embeddings](course_3/week2)
  * [Week 3: Sequence models](course_3/week3)
  * [Week 4: Sequence models and literature](course_3/week4)
* [Course 4: Sequences, Time Series and Prediction](course_4)
  * [Week 1: Time series examples](course_4/week1)
  * [Week 2: Deep Neural Networks for Time Series](course_4/week2)
  * [Week 3: Recurrent Neural Networks for Time Series](course_4/week3)
  * [Week 4: Real-world time series data](course_4/week4)

# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

第一周先從最基礎的 Tensorflow tools & syntax 學起

1. 從建立最簡單的神經網路開始來寫一個 regression
2. 使用神經網路來解 Computer Vision 問題! 實作 MNIST 和 FASHION MNIST 這兩個辨識數字和衣服的 dataset
3. 進階使用 CNN 的架構來建立神經網路模型
4. 最後用 CNN 來辨識真實世界的圖片

深入筆記在這邊: [Course 1: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning](course_1)

# Convolutional Neural Networks in TensorFlow

第二周將更深入的了解 Tensorflow 中如何使用 CNN 模型來解決電腦視覺問題

1. 處理更加雜亂的真實世界圖片 (圖片有不同大小、顏色)
2. 解決模型 overfitting 的問題: 利用 augmentation 和 dropout
3. 使用 transfer learning 來獲得神力、加速學習
4. 學會將 CNN 應用於 Multiclass classification

深入筆記在這邊: [Course 2: Convolutional Neural Networks in TensorFlow](course_2)

# Natural Language Processing in TensorFlow

第三周我們從原本的圖片，轉換跑道來處理文字，學習如何表達文字、並將文字來餵入模型、甚至創作文學 !

1. 學習怎麼讓要餵入模型的文字之間有意義
2. 進行文字的預處理 (tokenize, padding, embedding)
3. 掌握各種 sequence model (RNN, LSTM, GRU)
4. 使用 sequence model 與 text generation 來創造出莎士比亞的詩歌

深入筆記在這邊: [Course 3: Natural Language Processing in TensorFlow](course_3)

# Sequences, Time Series and Prediction

第四周將介紹什麼是 time series，從非 ML 的統計方式到 DL 方式來處理 time series

1. Time series 介紹、用統計學模型來預測走向
2. 處理 time series 的資料以便丟進模型訓練，用最簡單的 DNN 來訓練並預測 time series
3. 用 RNN, LSTM 來處理 time series 問題，接觸 lambda layer
4. 加入 CNN 來處理 time series 問題，處理太陽黑子的真實資料集

深入筆記在這邊: [Course 4: Sequences, Time Series and Prediction](course_4)

# Bonus

我還想學 Attention, Transformer, GANs, Auto-Encoder 等進階 DL 技巧 !!!

進階教材:

* [aymericdamien / TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
* [dragen1860 / TensorFlow-2.x-Tutorials](https://github.com/dragen1860/TensorFlow-2.x-Tutorials)
* [ageron / handson-ml2](https://github.com/ageron/handson-ml2)