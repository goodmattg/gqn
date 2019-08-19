"""
Contains (hyper-)parameters of the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.file import load_training_config_file

# GQN_DEFAULT_CONFIG is default.yaml loaded into DotMap
GQN_DEFAULT_CONFIG = load_training_config_file("config/default.yaml").model

# TODO: Write in a utility to override config given another yaml config
# i.e. YAML intersection. The whole command line override things isn't robust
