from __future__ import division, print_function
from tensorflow import keras as k
import numpy as np


L = k.layers

# random data for now
x_train = np.random.rand(100, 100, 100, 2)  # 100 images, shape (57, 57), channels static & moving
y_train = np.random.rand(100, 3) * 2 - 1  # 100 3-vectors

# normalisation: change dtype (speed)
x_train = x_train.astype('float32')
x_train /= x_train.std(axis=(1, 2))[:, None, None]  # ensures unit stdev

input_x = L.Input(x_train.shape[1:])

# 3 x 3 x (previous num channels = 2) kernels (32 times => 32 output channels)
# padding='same' for zero padding
x = L.Conv2D(32, 3, activation='relu', padding='same')(input_x)
x = L.Conv2D(64, 3, activation='relu', padding='same')(x)  # 3 x 3 x (previous num channels = 32) kernels (64 times)
x = L.MaxPooling2D(2)(x)  # downsample by factor of 2, using "maximum" interpolation
x = L.Dropout(0.25)(x)  # discard 25% outputs
x = L.Flatten()(x)  # vectorise
x = L.Dense(128, activation='relu')(x)  # traditional neural layer with 128 outputs
x = L.Dropout(0.5)(x)
x = L.Dense(3, activation=k.activations.tanh)(x)  # 3 outputs

model = k.Model(input_x, x)
model.summary()

# N x 3

# losses:  K.losses.*
# optimisers: K.optimizers.*
model.compile(optimizer=k.optimizers.Adam(), loss=k.losses.mean_squared_error)

model.fit(x_train, y_train, epochs=100, verbose=1)

loss = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', loss)
