"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .gqn_utils import broadcast_pose


def tower_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="TowerEncoder"):
    """
    Feed-forward convolutional architecture.
    """
    endpoints = {}

    net = tf.nn.conv2d(frames, filter=[256, 256, 2, 2], strides=2, padding="VALID")
    net = tf.nn.relu(net)

    skip1 = tf.nn.conv2d(net, filter=[128, 128, 1, 1], strides=1, padding="SAME")

    net = tf.nn.conv2d(net, filter=[128, 128, 3, 3], strides=1, padding="SAME")
    net = tf.nn.relu(net)

    # TODO(goodmattg): correct implementation for the skip connection?
    net = net + skip1

    net = tf.nn.conv2d(net, filter=[256, 256, 2, 2], strides=2, padding="VALID")
    net = tf.nn.relu(net)

    # tile the poses to match the embedding shape
    height, width = tf.shape(input=net)[1], tf.shape(input=net)[2]
    poses = broadcast_pose(poses, height, width)

    # concatenate the poses with the embedding
    net = tf.concat([net, poses], axis=3)

    skip2 = tf.nn.conv2d(net, filter=[128, 128, 1, 1], strides=1, padding="SAME")

    net = tf.nn.conv2d(net, filter=[128, 128, 3, 3], strides=1, padding="SAME")
    net = tf.nn.relu(net)

    # TODO(goodmattg): correct implementation for the skip connection?
    net = net + skip2

    net = tf.nn.conv2d(net, filter=[256, 256, 3, 3], strides=1, padding="SAME")
    net = tf.nn.relu(net)
    net = tf.nn.conv2d(net, filter=[256, 256, 1, 1], strides=1, padding="SAME")
    net = tf.nn.relu(net)

    return net, endpoints


def pool_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PoolEncoder"):
    """
  Feed-forward convolutional architecture with terminal global pooling.
  """
    net, endpoints = tower_encoder(frames, poses, scope)
    net = tf.reduce_mean(input_tensor=net, axis=[1, 2], keepdims=True)

    return net, endpoints

