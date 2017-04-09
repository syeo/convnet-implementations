import gc

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras import backend as K
import numpy as np


def CrossChannelNormalization(k=2, n=5, alpha=1e-4, beta=0.75):
    def f(X):
        _, R, C, CH = X.shape.as_list()
        half = n // 2
        squared = K.square(X)
        scales = []
        for i in range(CH):
            ch_from = max(0, i - half)
            ch_to = min(CH, i + half)
            squared_sum = (k + alpha * K.sum(squared[:, :, :, ch_from:ch_to], axis=-1)) ** beta
            scales.append(squared_sum)

        scale = K.stack(scales, axis=-1)
        return X / scale

    return Lambda(f)


def SplitTensor(num_splits, id_split):
    def f(X):
        assert X.shape.as_list()[-1] % num_splits == 0

        div = X.shape.as_list()[-1] // num_splits

        start = id_split * div
        end = start + div

        return X[:, :, :, start:end]

    return Lambda(f)


def get_second_layer_output(input_tensor):
    processed_input_tensor = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3))(input_tensor)

    return (
        SplitTensor(2, 0)(processed_input_tensor),
        SplitTensor(2, 1)(processed_input_tensor),
    )


def get_third_layer_output(input_tensors):
    output_tensors = []

    for (index, input_tensor) in enumerate(input_tensors):
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

    for (index, layer_func) in enumerate(layer_funcs):
        result = layer_func(result)

    return Model(inputs=inputs, outputs=[result])


if __name__ == '__main__':
    model = make_model()
    gc.collect()
