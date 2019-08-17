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

# def create_gqn_config(custom_params: dict):
#     """
#   Customizes the default parameters of the GQN with the parameters specified in
#   'custom_params'.

#   Returns:
#     GQNConfig
#   """
#     customized_keys = list(
#         set(custom_params.keys()).intersection(set(GQN_DEFAULT_PARAM_DICT.keys()))
#     )
#     customized_params = copy.deepcopy(GQN_DEFAULT_PARAM_DICT)
#     for k in customized_keys:
#         customized_params[k] = custom_params[k]
#     # return GQNConfig(**customized_params)
#     return
