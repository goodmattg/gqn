from argparse import ArgumentParser

from utils.argparse import *


def add_gqn_specific_arguments(parser: ArgumentParser) -> ArgumentParser:

    gqn_args = parser.add_argument_group("GGNN example specific arguments")

    gqn_args.add_argument(
        "--dataset",
        type=str,
        default="rooms_ring_camera",
        help="The name of the GQN dataset to use. \
        Available names are: \
        jaco | mazes | rooms_free_camera_no_object_rotations | \
        rooms_free_camera_with_object_rotations | rooms_ring_camera | \
        shepard_metzler_5_parts | shepard_metzler_7_parts",
    )

    # model parameters
    gqn_args.add_argument(
        "--seq_length",
        type=int,
        default=8,
        help="The number of generation steps of the DRAW LSTM.",
    )
    gqn_args.add_argument(
        "--context_size", type=int, default=5, help="The number of context points."
    )
    gqn_args.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Height and width of the squared input images.",
    )
    # solver parameters
    gqn_args.add_argument(
        "--adam_lr_alpha",
        type=float,
        default=5 * 10e-5,
        help="The initial learning rate of the ADAM solver.",
    )
    gqn_args.add_argument(
        "--adam_lr_beta",
        type=float,
        default=5 * 10e-6,
        help="The final learning rate of the ADAM solver.",
    )
    gqn_args.add_argument(
        "--anneal_lr_tau",
        type=int,
        default=1600000,
        help="The interval over which to anneal the learning rate from lr_alpha to \
        lr_beta.",
    )

    return parser
