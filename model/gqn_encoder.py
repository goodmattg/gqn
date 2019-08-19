"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Concatenate
from .gqn_utils import broadcast_pose


def gqn_encoder(frame_shape, pose_shape, pool=False):

    frame_input = Input(shape=frame_shape, name="frame_input")
    pose_input = Input(shape=pose_shape, name="pose_input")

    x = Conv2D(
        filters=256, kernel_size=2, strides=2, padding="valid", activation="relu"
    )(frame_input)

    skip1 = Conv2D(
        filters=128,
        kernel_size=1,
        strides=2,
        padding="same",
        activation=None,
        name="skip1",
    )(frame_input)

    x = Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same", activation="relu"
    )(x)

    x = Add()([x, skip1])

    x = Conv2D(
        filters=256, kernel_size=2, strides=2, padding="valid", activation="relu"
    )(x)

    # tile the poses to match the embedding shape
    height, width = x.shape[1], x.shape[2]
    poses = broadcast_pose(pose_input, height, width)

    x = Concatenate(axis=3)([x, poses])

    skip2 = Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding="same",
        activation=None,
        name="skip2",
    )(x)

    x = Conv2D(
        filters=128, kernel_size=3, strides=1, padding="same", activation="relu"
    )(x)

    x = Conv2D(
        filters=256, kernel_size=3, strides=1, padding="same", activation="relu"
    )(x)

    x = Add()([x, skip2])

    x = Conv2D(
        filters=256, kernel_size=3, strides=1, padding="same", activation="relu"
    )(x)

    x = Conv2D(
        filters=256,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        name="out",
    )(x)

    if pool:
        out = tf.keras.backend.mean(x, axis=[1, 2], keepdims=True)
    else:
        out = x

    return Model(inputs=[frame_input, pose_input], outputs=out)
