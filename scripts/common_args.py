from argparse import ArgumentParser

from utils.file import load_training_config_file
from utils.argparse import *

AVAILABLE_ACTIONS = ["train", "check", "test", "evaluate"]


def add_common_arguments(parser: ArgumentParser) -> ArgumentParser:

    core = parser.add_argument_group("core arguments")

    core.add_argument(
        "action",
        choices=AVAILABLE_ACTIONS,
        help="Take an action with your model (e.g. 'train')",
    )

    core.add_argument(
        "--model",
        type=str,
        help="Identifier of model to work with. Overrides config defined model",
    )

    core.add_argument(
        "--config_file",
        type=load_training_config_file,
        default="config/default.yaml",
        help="Configuration file absolute path",
    )

    core.add_argument(
        "--log_dir",
        type=dir_exists_write_privileges,
        default="logs",
        help="Log file storage directory path",
    )

    core.add_argument(
        "--data_dir",
        type=dir_exists_read_privileges,
        default="data",
        help="Data file storage directory path",
    )

    core.add_argument(
        "--restore_weights",
        type=file_exists,
        help="Restore model with pre-trained weights",
    )

    # training parameters
    core.add_argument(
        "--train_epochs", type=int, default=2, help="The number of epochs to train."
    )
    # snapshot parameters
    core.add_argument(
        "--chkpt_steps",
        type=int,
        default=10000,
        help="Number of steps between checkpoint saves.",
    )
    # memory management
    core.add_argument(
        "--batch_size", type=int, default=4, help="The number of data points per batch."
    )
    core.add_argument(
        "--memcap",
        type=float,
        default=1.0,
        help="Maximum fraction of memory to allocate per GPU.",
    )
    # data loading
    core.add_argument(
        "--queue_threads",
        type=int,
        default=4,
        help="How many parallel threads to run for data queuing.",
    )

    # FIXME: this should be called dataset prefetch buffer
    core.add_argument(
        "--queue_buffer", type=int, default=4, help="How many batches to queue up."
    )

    core.add_argument(
        "--log_steps", type=int, default=100, help="Global steps between log output."
    )

    core.add_argument("--freeze_graph_model", default=False, action="store_true")

    core.add_argument(
        "--initial_eval",
        default=False,
        action="store_true",
        help="Runs an evaluation before the first training iteration.",
    )

    core.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Enables debugging mode for more verbose logging and tensorboard output.",
    )

    core.add_argument(
        "--override",
        "-o",
        nargs="*",
        help="Override key value pairs in training configuration file for ad-hoc testing",
    )

    return parser
