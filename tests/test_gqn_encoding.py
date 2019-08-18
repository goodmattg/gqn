"""
Quick test script to shape-check graph definition of GQN encoder with random
toy data.
"""

import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import numpy as np

from model.gqn_params import GQN_DEFAULT_CONFIG
from model.gqn_encoder import gqn_encoder

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = 5
_DIM_POSE = GQN_DEFAULT_CONFIG.POSE_CHANNELS
_DIM_H_IMG = GQN_DEFAULT_CONFIG.IMG_HEIGHT
_DIM_W_IMG = GQN_DEFAULT_CONFIG.IMG_WIDTH
_DIM_C_IMG = GQN_DEFAULT_CONFIG.IMG_CHANNELS
_DIM_C_ENC = GQN_DEFAULT_CONFIG.ENC_CHANNELS


# random input

frame_shape = (_DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)
pose_shape = (_DIM_POSE,)

context_poses = np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE)

context_frames = np.random.rand(
    _BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG
)

# reshape context pairs into pseudo batch for representation network
context_poses_packed = np.reshape(context_poses, [-1, _DIM_POSE])
context_frames_packed = np.reshape(
    context_frames, [-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG]
)

tower_encoder = gqn_encoder(frame_shape, pose_shape, pool=False)
pool_encoder = gqn_encoder(frame_shape, pose_shape, pool=True)

# Tower Encoder
r_t_encoder_batch = tower_encoder.predict([context_frames_packed, context_poses_packed])

r_t_encoder = tf.reduce_sum(
    r_t_encoder_batch, axis=0
)  # add scene representations per data tuple

# Pool Encoder
r_p_encoder_batch = pool_encoder.predict([context_frames_packed, context_poses_packed])

r_p_encoder_batch = tf.reshape(
    r_p_encoder_batch, shape=[_BATCH_SIZE, _CONTEXT_SIZE, 1, 1, _DIM_C_ENC]
)  # 1, 1 for pool encoder only!

r_p_encoder = tf.reduce_sum(
    r_p_encoder_batch, axis=1
)  # add scene representations per data tuple

# Optional model visualization
# from tensorflow.keras.utils import plot_model
# plot_model(tower_encoder, to_file="model.png")

print("\nInputs:")
print("Frame sample shape: {}".format(frame_shape))
print("Frame batch shape: {}".format(context_frames.shape))
print("Packed frame batch shape: {}\n".format(context_frames_packed.shape))

print("Pose sample shape")
print("Pose sample shape: {}".format(pose_shape))
print("Pose batch shape: {}".format(context_poses.shape))
print("Packed pose batch shape: {}\n".format(context_poses_packed.shape))

print("Tower Encoder:")
print("Encoder output: {}".format(r_t_encoder_batch.shape))
print("Sum across scene repr. output: {}\n".format(r_t_encoder.shape))

print("Pool Encoder:")
print("Encoder output: {}".format(r_p_encoder_batch.shape))
print("Sum across scene repr. output: {}\n".format(r_p_encoder.shape))

print("TEST PASSED!")
