import argparse

import os
import sys
import yaml
import pickle as pkl

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

from dotmap import DotMap
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError


def pickle_loader(filepath, encoding=None):
    """Load pickle file"""
    with file_io.FileIO(filepath, mode="rb") as stream:
        if encoding:
            return pkl.load(stream, encoding=encoding)
        else:
            return pkl.load(stream)


def yaml_loader(filepath, use_dotmap=True):
    """Load a yaml file into a dictionary. Optionally wrap with DotMap."""
    with file_io.FileIO(filepath, mode="r") as stream:
        if use_dotmap:
            return DotMap(yaml.safe_load(stream))
        else:
            return yaml.safe_load(stream)


# Base Utilities (standard to boilerplate repository)
def load_file(filepath, load_func, **kwargs):
    """Generic file loader with missing/error messages."""
    try:
        print("Loading file from: {0}".format(filepath))
        return load_func(filepath, **kwargs)
    except NotFoundError as e:
        raise NotFoundError("File not found: {0}".format(filepath))
    except Exception as e:
        raise Exception("Unable to load file: {0}".format(filepath))


def load_training_config_file(filename):
    """Load a training configuration yaml file into a DotMap dictionary."""
    print("Loading training configuration file: {0}".format(filename))

    config_file_path = os.path.join(os.getcwd(), filename)
    return load_file(config_file_path, yaml_loader)


def load_data_pickle_file(filename, encoding=None):
    """Load a data pickle file/"""
    print("Loading pickled data file: {0}".format(filename))
    data_file_path = os.path.join(os.getcwd(), filename)
    return load_file(data_file_path, pickle_loader, encoding=encoding)
