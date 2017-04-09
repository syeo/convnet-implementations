import gc

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras import backend as K
import numpy as np


def CrossChannelNormalization(k=2, n=5, alpha=1e-4, beta=0.75):
    def f(X):
        num_channels = X.shape.as_list()[-1]
        half = n // 2
        squared = K.square(X)
        scales = []
        for i in range(num_channels):
            ch_from = max(0, i - half)
            ch_to = min(num_channels, i + half)
            squared_sum = (k + alpha * K.sum(squared[:, :, :, ch_from:ch_to], axis=-1)) ** beta
            scales.append(squared_sum)

        scale = K.stack(scales, axis=-1)
        return X / scale

    return Lambda(f)


def get_second_layer_output(input_tensor):
    return (
        Conv2D(48, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3))(input_tensor),
        Conv2D(48, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3))(input_tensor),
    )


def get_third_layer_output(input_tensors):
    output_tensors = []

    for input_tensor in input_tensors:
        processed_input_tensor = MaxPooling2D((3, 3), (2, 2))(input_tensor)
        processed_input_tensor = CrossChannelNormalization()(processed_input_tensor)
        processed_input_tensor = ZeroPadding2D((2, 2))(processed_input_tensor)
        processed_input_tensor = Conv2D(128, (5, 5), activation='relu')(processed_input_tensor)
        output_tensors.append(processed_input_tensor)

    return tuple(output_tensors)


def get_fourth_layer_output(input_tensors):
    processed_input_tensor = Concatenate()(list(input_tensors))
    processed_input_tensor = MaxPooling2D((3, 3), (2, 2))(processed_input_tensor)
    processed_input_tensor = CrossChannelNormalization()(processed_input_tensor)
    processed_input_tensor = ZeroPadding2D()(processed_input_tensor)

    return (
        Conv2D(192, (3, 3), activation='relu')(processed_input_tensor),
        Conv2D(192, (3, 3), activation='relu')(processed_input_tensor),
    )


def get_fifth_layer_output(input_tensors):
    output_tensors = []

    for input_tensor in input_tensors:
        output_tensors.append(Conv2D(192, (3, 3), activation='relu')(ZeroPadding2D()(input_tensor)))

    return tuple(output_tensors)


def get_sixth_layer_output(input_tensors):
    output_tensors = []

    for input_tensor in input_tensors:
        output_tensors.append(Conv2D(128, (3, 3), activation='relu')(ZeroPadding2D()(input_tensor)))

    return tuple(output_tensors)


def get_seventh_layer_output(input_tensors):
    processed_input_tensor = Concatenate()(list(input_tensors))
    processed_input_tensor = Flatten()(processed_input_tensor)
    processed_input_tensor = Dropout(0.5)(processed_input_tensor)

    return Dense(4096, activation='relu')(processed_input_tensor)


def get_eighth_layer_output(input_tensor):
    processed_input_tensor = Dropout(0.5)(input_tensor)

    return Dense(4906, activation='relu')(Dropout(0.5)(processed_input_tensor))


def get_ninth_layer_output(input_tensor):
    return Dense(1000, activation='softmax')(input_tensor)


def make_model():
    inputs = Input(shape=(227, 227, 3))

    layer_funcs = [
        get_second_layer_output,
        get_third_layer_output,
        get_fourth_layer_output,
        get_fifth_layer_output,
        get_sixth_layer_output,
        get_seventh_layer_output,
        get_eighth_layer_output,
        get_ninth_layer_output,
    ]

    result = inputs

    for layer_func in layer_funcs:
        result = layer_func(result)

    return Model(inputs=inputs, outputs=result)


if __name__ == '__main__':
    model = make_model()
    gc.collect()
