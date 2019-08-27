"""
Test script to shape-check graph definition of GQN latent space
inference with random toy data.
"""

import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, ".."))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import numpy as np

from model.gqn_params import GQN_DEFAULT_CONFIG
from model.gqn_rnn import GeneratorLSTMCell, generator_rnn
from model.gqn_utils import broadcast_pose

# config
gqn_config = GQN_DEFAULT_CONFIG.copy()
img_size = 128

gqn_config.IMG_HEIGHT = img_size
gqn_config.IMG_WIDTH = img_size
gqn_config.ENC_HEIGHT = img_size // 4  # must be 1/4 of target frame height
gqn_config.ENC_WIDTH = img_size // 4  # must be 1/4 of target frame width

_BATCH_SIZE = 5
_DIM_POSE = gqn_config.POSE_CHANNELS
_DIM_H_IMG = gqn_config.IMG_HEIGHT
_DIM_W_IMG = gqn_config.IMG_WIDTH
_DIM_C_IMG = gqn_config.IMG_CHANNELS
_DIM_R_H = gqn_config.ENC_HEIGHT
_DIM_R_W = gqn_config.ENC_WIDTH
_DIM_R_C = gqn_config.ENC_CHANNELS
_SEQ_LENGTH = gqn_config.SEQ_LENGTH

# Dummy data
query_poses = np.random.rand(_BATCH_SIZE, _DIM_POSE)
target_frame = np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)
scene_representations = np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C)

# import pdb

# pdb.set_trace()

tf.config.experimental_run_functions_eagerly(True)
rnn = generator_rnn(scene_representations, query_poses, gqn_config)

