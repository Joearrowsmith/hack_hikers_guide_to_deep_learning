import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


"""
Load the dataset
"""

dataset = keras.datasets.mnist # http://yann.lecun.com/exdb/mnist/

(x_train, y_train) , (x_test, y_test) = dataset.load_data()

print("data shape: ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print("data type: ", type(x_train))

"""
Preprocessing
"""

def normalise_train_data(data):
    data = data / 255
    """
    we will be using a cnn so we need to add an additional dimension to our data; this dimension is what you would consider to typically be the colour channel. 
    For grayscale images it is of size 1.
    """
    data = np.reshape(data, (*data.shape, 1))
    return data

print(x_train.shape, x_test.shape)

x_train = normalise_train_data(x_train)
x_test = normalise_train_data(x_test)

print(x_train.shape, x_test.shape)

num_classes = y_train.max() + 1 # +1 due to zero index
assert num_classes == 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

assert x_train.max() <= 1, f"x_train max value is larger than expected: {x_train.max()} !<= 1"
assert x_train.min() >= 0, f"x_train min value is smaller than expected: {x_train.min()} !>= 0"
assert x_test.max() <= 1, f"x_test max value is larger than expected: {x_test.max()} !<= 1"
assert x_test.min() >= 0, f"x_test min value is smaller than expected: {x_test.min()} !>= 0"
assert y_train.shape[1] == 10
assert y_test.shape[1] == 10

""" we should also create a validation dataset from our train data """
train_valid_split = 0.90
tr_val_idx = int(x_train.shape[0] * train_valid_split)
x_train, x_val = x_train[:tr_val_idx], x_train[tr_val_idx:]
y_train, y_val = y_train[:tr_val_idx], y_train[tr_val_idx:]

input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:][0]


print(input_shape, output_shape)
assert x_train.shape[1:] == x_val.shape[1:] == x_test.shape[1:], "x data is not in the same formats"
assert y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:], "x data is not in the same formats"
assert input_shape == (28, 28, 1), "Input shape is not in the expected format"
assert output_shape == 10, "Output shape is not in the expected format"

""" lets try out several different architectures: """

""" cnn model #1 """

cnn_1 = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" cnn model #2 """

cnn_2 = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape),
    keras.layers.MaxPool2D(),
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" cnn model #3 """

cnn_3 = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=input_shape),
    keras.layers.Dropout(0.1),                    
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Conv2D(32, kernel_size=(3, 3),
                        activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(output_shape, activation='softmax')
])



""" lets store our models in a dict (a list would also be valid) so we can loop through them """ 

models = {
    "single_cnn" : [cnn_1, 50],
    "deep_cnn" : [cnn_2, 50],
    "fancy_cnn" : [cnn_3, 50],
}

scores = []

for i in models:
    print("-------------------------------")
    print("model currently looking at: ", i)

    model, e = models[i]
    print("training with epochs: ", e)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=16,
                     epochs=e,
                     verbose=1,
                     validation_data=(x_val, y_val))
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])

    scores.append([hist, i, score[1]])

print("-------------------------------")

for i in scores:
    print(f"{i[1]}: {i[2]*100:.1f}%")
    plt.plot(i[0].history['accuracy'], label=f'{i[1]} train acc')
    plt.plot(i[0].history['val_accuracy'], label=f'{i[1]} val acc')

plt.legend()
plt.grid(True)
plt.show()