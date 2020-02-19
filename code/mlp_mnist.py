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

#plt.imshow(x_train[1])
#plt.show()

print("sample of what our data looks like (random row from an image): ", x_train[1, 14, :28])
print("bounds:", x_train[0].max(), x_train[0].min())

"""
We are working with grayscale images. These images have 1 colour channel and generally the value varies from 0 to 255.
Due to gradient flows in the backpropagation algorithm, neural networks typically need data to be in the range: 0 to 1, or -1 to 1, or a zscore distribution.
In this case we will divide the vaues in our images by 255 to scale the input values to be between 0 and 1.

Critically, we need to apply the same transformation to our testdata, but we can't peak at our test data to determine how we should transform it. 
We will explore this in more detail later.
"""

def normalise_train_data(data):
    data = data / 255
    return data
 
x_train = normalise_train_data(x_train)
x_test = normalise_train_data(x_test)

"""
Lets look at our data again:
"""

#plt.imshow(x_train[1])
#plt.show()

print("sample of what our data looks like (random row from an image): ", x_train[1, 14, :28])
print("bounds:", x_train[0].max(), x_train[0].min())

print("y (target) output: ", y_train[0])

"""
We need to convert our target values to a categorical output
"""

num_classes = y_train.max() + 1 # +1 due to zero index
assert num_classes == 10
print("train num classes: ", num_classes)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Lets check we haven't made any mistakes
"""

assert x_train.max() <= 1, f"x_train max value is larger than expected: {x_train.max()} !<= 1"
assert x_train.min() >= 0, f"x_train min value is smaller than expected: {x_train.min()} !>= 0"
assert x_test.max() <= 1, f"x_test max value is larger than expected: {x_test.max()} !<= 1"
assert x_test.min() >= 0, f"x_test min value is smaller than expected: {x_test.min()} !>= 0"

assert y_train.shape[1] == 10
assert y_test.shape[1] == 10

""" we should also create a validation dataset from our train data """

train_valid_split = 0.90
tr_val_idx = int(x_train.shape[0] * train_valid_split)

print("tr_val_idx: ", tr_val_idx)

x_train, x_val = x_train[:tr_val_idx], x_train[tr_val_idx:]
y_train, y_val = y_train[:tr_val_idx], y_train[tr_val_idx:]

print("x_train, x_val: ", x_train.shape, x_val.shape)
print("y_train, y_val: ", y_train.shape, y_val.shape)

"""
Great, we have prepared our data, now what? 
Lets build a simple network!

There are many ways to define a model in tensorflow, lets go with the simplest way: keras.sequential
another method worth looking at when you understand the sequential framework is the functional framework. https://www.tensorflow.org/tutorials/customization/custom_layers
"""

input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:][0]

print(input_shape, output_shape)

assert x_train.shape[1:] == x_val.shape[1:] == x_test.shape[1:], "x data is not in the same formats"
assert y_train.shape[1:] == y_val.shape[1:] == y_test.shape[1:], "x data is not in the same formats"
assert input_shape == (28, 28), "Input shape is not in the expected format"
assert output_shape == 10, "Output shape is not in the expected format"


""" lets try out several different architectures: """

""" mlp model #1 """

no_hidden_layer_network = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" mlp model #2 """

single_mlp = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"), # tanh can also be used however relu is typically the best. (Don't worry about this for now.) 
    keras.layers.Dense(output_shape, activation='softmax')
])

""" mlp model #3 """

single_dropout_mlp = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" mlp model #4 """

double_dropout_mlp = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" mlp model #5 """

deeper_dropout_mlp_with_batchnorm = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(32, activation="relu"), 
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(output_shape, activation='softmax')
])

""" lets store our models in a dict (a list would also be valid) so we can loop through them """ 

models = {
    "no_hidden_layers" : no_hidden_layer_network,
    "single_hidden_layer" : single_mlp,
    "single_hidden_layer_with_dropout" : single_dropout_mlp,
    "two_hidden_layers_with_dropout" : double_dropout_mlp,
    "three_hidden_layers_with_dropout_and_batchnorm" : deeper_dropout_mlp_with_batchnorm,
}

scores = []

for i in models:
    print("-------------------------------")
    print("model currently looking at: ", i)

    model = models[i]

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=10,
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
    plt.plot(i[0].history['val_accuracy'], '--', label=f'{i[1]} val acc')

plt.legend()
plt.show()

"""

These models are not good at this problem, why?














- answer: lack of spacial information
"""