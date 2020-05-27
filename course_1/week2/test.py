from tensorflow import keras

dataset = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

print(dataset.load_data())