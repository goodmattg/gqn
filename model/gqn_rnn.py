from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

from collections import namedtuple

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    ConvLSTM2D,
    Conv2DTranspose,
    Add,
    Concatenate,
    RNN,
)

from model.gqn_utils import broadcast_pose, sample_z

# Encoded image representation, query pose, z (FIXME: Removed sample "z" from input here)
_GeneratorCellInput = namedtuple("GeneratorCellInput", ["representation", "query_pose"])
# canvas, hidden state (h)
_GeneratorCellOutput = namedtuple("GeneratorCellOutput", ["canvas", "lstm"])
# canvas, (new canvas ("c"), h)
_GeneratorCellState = namedtuple("GeneratorCellState", ["canvas", "lstm_c", "lstm_h"])


class GeneratorLSTMCell(tf.keras.layers.Layer):
    def __init__(
        self,
        input_shape,
        output_channels,
        canvas_channels,
        kernel_size=5,
        use_bias=True,
        forget_bias=True,
        name="GeneratorLSTMCell",
    ):
        super(GeneratorLSTMCell, self).__init__(name=name)

        self._conv_lstm_2d = ConvLSTM2D(
            filters=4 * output_channels,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation=None,
            use_bias=use_bias,
            unit_forget_bias=forget_bias,
            name="DownsampleGeneratorOutput",
        )

        self._upsample_conv = Conv2DTranspose(
            filters=canvas_channels,
            kernel_size=4,
            strides=4,
            name="UpsampleGeneratorOutput",
        )

        # pdb.set_trace()

        canvas_shape = [4 * x for x in input_shape[:-1]] + [canvas_channels]
        lstm_output_size = input_shape[:-1] + [output_channels]

        self._output_size = _GeneratorCellOutput(
            tf.TensorShape(canvas_shape), tf.TensorShape(lstm_output_size)
        )
        self._state_size = _GeneratorCellState(
            tf.TensorShape(canvas_shape),
            tf.TensorShape(lstm_output_size),
            tf.TensorShape(lstm_output_size),
        )

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def build(self, input_shape):
        # FIXME: No weights are added here??
        # self.add_weight()???
        # self.built = True????
        pass

    @tf.function
    def sample(self):

        z = sample_z(self.state.lstm.h, scope="Sample_eta_pi")

    @tf.function
    def call(self, inputs: _GeneratorCellInput, state: _GeneratorCellState):

        import pdb

        pdb.set_trace()

        # FIXME: Remove scope
        # z = sample_z(state.lstm.h, scope="Sample_eta_pi")

        canvas, cell_state, hidden_state = state
        # As in, subsample?
        sub_output, new_sub_state = self._conv_lstm_2d(
            inputs, (cell_state, hidden_state)
        )
        new_canvas = canvas + self._upsample_conv(sub_output)

        # new_output = _GeneratorCellOutput(new_canvas, sub_output)
        # new_state = _GeneratorCellState(new_canvas, new_sub_state)

        return new_canvas, new_canvas


def generator_rnn(representations, query_poses, params):

    batch, height, width, channels = representations.shape
    query_poses = broadcast_pose(query_poses, height, width)

    cell = GeneratorLSTMCell(
        input_shape=[height, width, params.GENERATOR_INPUT_CHANNELS],
        output_channels=params.LSTM_OUTPUT_CHANNELS,
        canvas_channels=params.LSTM_CANVAS_CHANNELS,
        kernel_size=params.LSTM_KERNEL_SIZE,
        name="GeneratorCell",
    )

    # Define the generator RNN with unrolling
    # Input is representations and query_poses
    # Separate input is z (sampled)

    # We want this thing unrolled and the output at each time step (so we can see the canvas)
    repr_input = Input(shape=representations.shape, name="representations")
    query_poses_input = Input(shape=query_poses.shape, name="query_poses")
    # query_poses = broadcast_pose(query_poses_input, height, width)
    # merged_input = Concatenate()([repr_input, query_poses])
    named_input = _GeneratorCellInput(repr_input, query_poses_input)

    # pdb.set_trace()

    outputs = RNN(cell, unroll=True, return_state=True)(named_input)
    target_canvas = outputs[-1].canvas
    mu_target = eta_g(target_canvas, channels=params.IMG_CHANNELS, scope="eta_g")

    return mu_target

