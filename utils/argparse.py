import argparse
import os
import operator
import sys
from functools import reduce


def get_from_dict(dictionary, key_list):
    """Get value from dictionary with arbitrary depth via descending list of keys."""
    return reduce(operator.getitem, key_list, dictionary)


def set_in_dict(dictionary, key_list, value):
    """Set value from dictionary with arbitrary depth via descending list of keys."""
    get_from_dict(dictionary, key_list[:-1])[key_list[-1]] = value


def cast_to_type(type_abbrev, val):
    """Convert string input value to explicitly denoted type. Types are as follows:
    "f" -> float
    "i" -> integer
    "b" -> boolean
    "s" -> string
    """
    if type_abbrev == "f":
        return float(val)
    elif type_abbrev == "i":
        return int(val)
    elif type_abbrev == "b":
        return val.lower() in ("yes", "true", "t", "1")
    else:
        return val


def override_dotmap(overrides, config):
    """Override DotMap dictionary with explicitly typed values."""
    for i in range(len(overrides) // 3):
        key, type_abbrev, val = overrides[i * 3 : (i + 1) * 3]
        set_in_dict(config, key.split("."), cast_to_type(type_abbrev, val))


def file_exists(prospective_file):
    """Check if the prospective file exists"""
    file_path = os.path.join(os.getcwd(), prospective_file)
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError("File: '{0}' does not exist".format(file_path))
    return file_path


def dir_exists_write_privileges(prospective_dir):
    """Check if the prospective directory exists with write privileges."""
    dir_path = os.path.join(os.getcwd(), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.W_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not writable".format(dir_path)
        )
    return dir_path


def dir_exists_read_privileges(prospective_dir):
    """Check if the prospective directory exists with read privileges."""
    dir_path = os.path.join(os.getcwd(), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.R_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not readable".format(dir_path)
        )
    return dir_path
