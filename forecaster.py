import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from RNN.conv_gru import ConvGRUCell
from RNN.conv_stlstm import ConvSTLSTMCell
from RNN.PredRNN import PredRNNCell
from config import c
from config import config_gru_fms, config_deconv_infer
from tf_utils import deconv2d_act


class Forecaster(object):
    def __init__(self, batch, seq, gru_filter, gru_in_chanel,
                 deconv_kernel, deconv_stride, h2h_kernel,
                 i2h_kernel, rnn_states, height, width):
        if c.DTYPE == "single":
            self._dtype = tf.float32
        elif c.DTYPE == "HALF":
            self._dtype = tf.float16

        self._batch = batch
        self._seq = seq
        self._h = height
        self._w = width

        self.stack_num = len(gru_filter)
        self.rnn_blocks = []
        self.rnn_states = rnn_states
        self.conv_kernels = []
        self.conv_bias = []
        self.conv_stride = deconv_stride
        self.infer_shape = []
        self._infer_shape = config_deconv_infer(height, deconv_stride)

        self.final_conv = []
        self.final_bias = []

        self._gru_fms = config_gru_fms(height, deconv_stride)
        self._gru_filter = gru_filter
        self._conv_fms = deconv_kernel
        self._gru_in_chanel = gru_in_chanel
        self._h2h_kernel = h2h_kernel
        self._i2h_kernel = i2h_kernel

        self.pred = []

        self.build_rnn_blocks()
        self.init_parameters()

    def build_rnn_blocks(self):
        """
        same as encoder
        first rnn input (b, 180, 180 ,192) output (b, 180, 180, 64)
        :return:
        """
        with tf.variable_scope("Forecaster"):
            for i in range(len(self._gru_fms)):
                if c.RNN_CELL == "conv_gru":
                    cell = ConvGRUCell(num_filter=self._gru_filter[i],
                                       b_h_w=(self._batch,
                                              self._gru_fms[i],
                                              self._gru_fms[i]),
                                       h2h_kernel=self._h2h_kernel[i],
                                       i2h_kernel=self._i2h_kernel[i],
                                       name="e_cgru_" + str(i),
                                       chanel=self._gru_in_chanel[i])
                elif c.RNN_CELL == "st_lstm":
                    cell = ConvSTLSTMCell(num_filter=self._gru_filter[i],
                                          b_h_w=(self._batch,
                                                 self._gru_fms[i],
                                                 self._gru_fms[i]),
                                          kernel=self._i2h_kernel[i],
                                          name="e_stlstm_" + str(i),
                                          chanel=self._gru_in_chanel[i])
                elif c.RNN_CELL == "PredRNN":
                    cell = PredRNNCell(num_filter=self._gru_filter[i],
                                       b_h_w=(self._batch,
                                              self._gru_fms[i],
                                              self._gru_fms[i]),
                                       kernel=self._i2h_kernel[i],
                                       name="e_predrnn_" + str(i),
                                       chanel=self._gru_in_chanel[i],
                                       layers=c.PRED_RNN_LAYERS)
                else:
                    raise NotImplementedError

                self.rnn_blocks.append(cell)

    def init_parameters(self):
        with tf.variable_scope("Forecaster", auxiliary_name_scope=False):
            if c.UP_SAMPLE_TYPE == "deconv":
                for i in range(len(self._conv_fms)):
                    self.conv_kernels.append(tf.get_variable(name=f"Deconv{i}_W",
                                                             shape=self._conv_fms[i],
                                                             initializer=xavier_initializer(uniform=False),
                                                             dtype=self._dtype))
                    self.conv_bias.append(tf.get_variable(name=f"Deconv{i}_b",
                                                          shape=[self._conv_fms[i][-2]],
                                                          initializer=tf.zeros_initializer))
                    if c.SEQUENCE_MODE:
                        self.infer_shape.append(
                            (self._batch, self._infer_shape[i], self._infer_shape[i], self._conv_fms[i][-2]))
                    else:
                        self.infer_shape.append(
                            (self._batch*self._seq, self._infer_shape[i], self._infer_shape[i], self._conv_fms[i][-2]))
            elif c.UP_SAMPLE_TYPE == "inception":
                for i in range(len(self._conv_fms)):
                    conv_kernels = []
                    biases = []
                    shapes = []
                    for j in range(len(self._conv_fms[i])):
                        kernel = self._conv_fms[i][j]
                        conv_kernels.append(tf.get_variable(name=f"Conv{i}_W{j}",
                                                            shape=kernel,
                                                            initializer=xavier_initializer(uniform=False),
                                                            dtype=self._dtype))
                        biases.append(tf.get_variable(name=f"Conv{i}_b{j}",
                                                      shape=kernel[-2],
                                                      initializer=tf.zeros_initializer))
                        if c.SEQUENCE_MODE:
                            shapes.append(
                                (self._batch, self._infer_shape[i], self._infer_shape[i], kernel[-2]))
                        else:
                            shapes.append((self._batch*self._seq, self._infer_shape[i], self._infer_shape[i], kernel[-2]))
                    self.conv_kernels.append(conv_kernels)
                    self.conv_bias.append(biases)
                    self.infer_shape.append(shapes)
            else:
                raise NotImplementedError

            self.final_conv.append(tf.get_variable(name="Final_conv1_W",
                                                   shape=(3, 3, 8, 8),
                                                   initializer=xavier_initializer(uniform=False),
                                                   dtype=self._dtype))
            self.final_bias.append(tf.get_variable(name="Final_conv1_b",
                                                   shape=[8],
                                                   initializer=tf.zeros_initializer))
            self.final_conv.append(tf.get_variable(name="Final_conv2",
                                                   shape=(1, 1, 8, 1),
                                                   initializer=xavier_initializer(uniform=False),
                                                   dtype=self._dtype))
            self.final_bias.append(tf.get_variable(name="Final_conv2_b",
                                                   shape=[1],
                                                   initializer=tf.zeros_initializer))

    def rnn_forecaster(self):
        with tf.variable_scope("Forecaster", auxiliary_name_scope=False, reuse=tf.AUTO_REUSE):
            in_data = None
            for i in range(self.stack_num-1, -1, -1):
                output, states = self.rnn_blocks[i].unroll(inputs=in_data,
                                                           length=self._seq,
                                                           begin_state=self.rnn_states[i])
                deconv = deconv2d_act(input=output,
                                      name=f"Deconv{i}",
                                      kernel=self.conv_kernels[i],
                                      bias=self.conv_bias[i],
                                      infer_shape=self.infer_shape[i],
                                      strides=self.conv_stride[i])

                in_data = deconv
            in_data = tf.reshape(in_data, [self._batch*self._seq, self._h, self._w, -1])
            conv_final = tf.nn.conv2d(in_data, self.final_conv[0], strides=(1, 1, 1, 1), padding="SAME",
                                      name="final_conv")
            conv_final = tf.nn.leaky_relu(tf.nn.bias_add(conv_final, self.final_bias[0]))
            pred = tf.nn.conv2d(conv_final, filter=self.final_conv[1], strides=(1, 1, 1, 1), padding="SAME",
                                name="Pred")
            pred = tf.nn.leaky_relu(tf.nn.bias_add(pred, self.final_bias[1]))
            pred = tf.reshape(pred, shape=(self._batch, self._seq, self._h, self._w, 1))
            self.pred = pred

    def rnn_forecaster_step(self):
        with tf.variable_scope("Forecaster", auxiliary_name_scope=False, reuse=tf.AUTO_REUSE):
            in_data = None
            for i in range(self.stack_num-1, -1, -1):
                output, states = self.rnn_blocks[i](inputs=in_data,
                                                    state=self.rnn_states[i])
                deconv = deconv2d_act(input=output,
                                      name=f"Deconv{i}",
                                      kernel=self.conv_kernels[i],
                                      bias=self.conv_bias[i],
                                      infer_shape=self.infer_shape[i],
                                      strides=self.conv_stride[i])

                in_data = deconv
            in_data = tf.reshape(in_data, [self._batch, self._h, self._w, -1])
            conv_final = tf.nn.conv2d(in_data, self.final_conv[0], strides=(1, 1, 1, 1), padding="SAME",
                                      name="final_conv")
            conv_final = tf.nn.leaky_relu(tf.nn.bias_add(conv_final, self.final_bias[0]))
            pred = tf.nn.conv2d(conv_final, filter=self.final_conv[1], strides=(1, 1, 1, 1), padding="SAME",
                                name="Pred")
            pred = tf.nn.leaky_relu(tf.nn.bias_add(pred, self.final_bias[1]))
            pred = tf.reshape(pred, shape=(self._batch, 1, self._h, self._w, 1))
            self.pred.append(pred)
