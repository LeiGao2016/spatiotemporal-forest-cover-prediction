import math
import numpy as np
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Add, Activation, Permute, multiply, Layer, Conv1D, \
    Conv2D, ReLU, SeparableConv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

class EcaBlockt1(Layer):
    def __init__(self, b=1.0, gama=2.0, data_format='channels_last'):
        super(EcaBlockt1, self).__init__()
        self.b = b
        self.gama = gama
        self._data_format = data_format

    def call(self, x):
        N, H, W, C = x.shape
        t = int(abs((math.log(C, 2) + self.b) / self.gama))
        k_size = t if t % 2 else t + 1
        squeeze = tf.reduce_mean(x, [1, 2])  # (5,32)
        squeeze = tf.expand_dims(squeeze, axis=1)  # (5,1,32)
        attn = Conv1D(filters=C, kernel_size=k_size, padding='same',
                      use_bias=False, data_format=self._data_format)(squeeze)
        attn = tf.expand_dims(attn, axis=2)
        attn = tf.math.sigmoid(attn)
        scale = x * attn
        return x * attn

class EcaBlockt2(Layer):
    def __init__(self, b=1.0, gama=2.0, data_format='channels_last'):
        super(EcaBlockt2, self).__init__()
        self.b = b
        self.gama = gama
        self._data_format = data_format

    def call(self, x):
        N, H, W, C = x.shape
        t = int(abs((math.log(C, 2) + self.b) / self.gama))
        k_size = t if t % 2 else t + 1
        squeeze = tf.reduce_mean(x, [1, 2])  # (5,32)
        squeeze = tf.expand_dims(squeeze, axis=1)  # (5,1,32)
        attn = Conv1D(filters=C, kernel_size=k_size, padding='same',
                      use_bias=False, data_format=self._data_format)(squeeze)
        attn = tf.expand_dims(attn, axis=2)
        attn = tf.math.sigmoid(attn)
        scale = x * attn
        return x * attn

class EcaBlockt3(Layer):
    def __init__(self, b=1.0, gama=2.0, data_format='channels_last'):
        super(EcaBlockt3, self).__init__()
        self.b = b
        self.gama = gama
        self._data_format = data_format

    def call(self, x):
        N, H, W, C = x.shape
        t = int(abs((math.log(C, 2) + self.b) / self.gama))
        k_size = t if t % 2 else t + 1
        squeeze = tf.reduce_mean(x, [1, 2])  # (5,32)
        squeeze = tf.expand_dims(squeeze, axis=1)  # (5,1,32)
        attn = Conv1D(filters=C, kernel_size=k_size, padding='same',
                      use_bias=False, data_format=self._data_format)(squeeze)
        attn = tf.expand_dims(attn, axis=2)
        attn = tf.math.sigmoid(attn)
        scale = x * attn
        return x * attn

class EcaBlockt4(Layer):
    def __init__(self, b=1.0, gama=2.0, data_format='channels_last'):
        super(EcaBlockt4, self).__init__()
        self.b = b
        self.gama = gama
        self._data_format = data_format

    def call(self, x):
        N, H, W, C = x.shape
        t = int(abs((math.log(C, 2) + self.b) / self.gama))
        k_size = t if t % 2 else t + 1
        squeeze = tf.reduce_mean(x, [1, 2])  # (5,32)
        squeeze = tf.expand_dims(squeeze, axis=1)  # (5,1,32)
        attn = Conv1D(filters=C, kernel_size=k_size, padding='same',
                      use_bias=False, data_format=self._data_format)(squeeze)
        attn = tf.expand_dims(attn, axis=2)
        attn = tf.math.sigmoid(attn)
        scale = x * attn
        return x * attn

class PAM1(Layer):
    """Position attention module"""

    def __init__(self):
        super(PAM1, self).__init__()
        self.gamma = tf.Variable(tf.ones(1))

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x1 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)  # (4, 10, 49, 32)
        x1_trans = tf.transpose(x1, perm=[0, 3, 1,
                                          2])
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c,
                                                       h * w])
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape,
                                              perm=[0, 2, 1])
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        x1_mutmul = tf.nn.softmax(x1_mutmul)
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x2_trans = tf.transpose(x2,
                                perm=[0, 3, 1, 2])
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c,
                                                       h * w])
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2,
                                                        1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans
        x2_mutmul = tf.reshape(x2_mutmul,
                               shape=[-1, c, h, w])
        x2_mutmul = tf.transpose(x2_mutmul,
                                 perm=[0, 2, 3, 1])
        x2_mutmul = x2_mutmul * self.gamma
        output = Add()([x2_mutmul, inputs])

        return output

class PAM2(Layer):
    """Position attention module"""

    def __init__(self):
        super(PAM2, self).__init__()
        self.gamma = tf.Variable(tf.ones(1))

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x1 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x1_trans = tf.transpose(x1, perm=[0, 3, 1,
                                          2])
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c,
                                                       h * w])
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape,
                                              perm=[0, 2, 1])
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        x1_mutmul = tf.nn.softmax(x1_mutmul)
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x2_trans = tf.transpose(x2,
                                perm=[0, 3, 1, 2])
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c,
                                                       h * w])
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2,
                                                        1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans
        x2_mutmul = tf.reshape(x2_mutmul,
                               shape=[-1, c, h, w])
        x2_mutmul = tf.transpose(x2_mutmul,
                                 perm=[0, 2, 3, 1])
        x2_mutmul = x2_mutmul * self.gamma
        output = Add()([x2_mutmul, inputs])
        return output

class PAM3(Layer):
    """Position attention module"""

    def __init__(self):
        super(PAM3, self).__init__()
        self.gamma = tf.Variable(tf.ones(1))

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x1 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x1_trans = tf.transpose(x1, perm=[0, 3, 1,
                                          2])
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c,
                                                       h * w])
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape,
                                              perm=[0, 2, 1])
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        x1_mutmul = tf.nn.softmax(x1_mutmul)
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x2_trans = tf.transpose(x2,
                                perm=[0, 3, 1, 2])
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c,
                                                       h * w])
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2,
                                                        1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans
        x2_mutmul = tf.reshape(x2_mutmul,
                               shape=[-1, c, h, w])
        x2_mutmul = tf.transpose(x2_mutmul,
                                 perm=[0, 2, 3, 1])
        x2_mutmul = x2_mutmul * self.gamma
        output = Add()([x2_mutmul, inputs])
        return output

class PAM4(Layer):
    """Position attention module"""

    def __init__(self):
        super(PAM4, self).__init__()
        self.gamma = tf.Variable(tf.ones(1))

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x1 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x1_trans = tf.transpose(x1, perm=[0, 3, 1,
                                          2])
        x1_trans_reshape = tf.reshape(x1_trans, shape=[-1, c,
                                                       h * w])
        x1_trans_reshape_trans = tf.transpose(x1_trans_reshape,
                                              perm=[0, 2, 1])
        x1_mutmul = x1_trans_reshape_trans @ x1_trans_reshape
        x1_mutmul = tf.nn.softmax(x1_mutmul)
        x2 = SeparableConv2D(filters=c, kernel_size=(1, 1), strides=1,
                             padding='same')(inputs)
        x2_trans = tf.transpose(x2,
                                perm=[0, 3, 1, 2])
        x2_trans_reshape = tf.reshape(x2_trans, shape=[-1, c,
                                                       h * w])
        x1_mutmul_trans = tf.transpose(x1_mutmul, perm=[0, 2,
                                                        1])
        x2_mutmul = x2_trans_reshape @ x1_mutmul_trans
        x2_mutmul = tf.reshape(x2_mutmul,
                               shape=[-1, c, h, w])
        x2_mutmul = tf.transpose(x2_mutmul,
                                 perm=[0, 2, 3, 1])
        x2_mutmul = x2_mutmul * self.gamma
        output = Add()([x2_mutmul, inputs])
        return output


def dual_attentiont1(x_input, in_dim):
    pam_layer = PAM1()
    cam_layer = EcaBlockt1()
    pam_output = pam_layer(x_input)
    cam_output = cam_layer(x_input)
    output_summed = tf.add(pam_output, cam_output)
    output_final = Conv2D(filters=in_dim, kernel_size=(3, 3), padding='same')(
        output_summed)
    return output_final


def dual_attentiont2(x_input, in_dim):
    # print(np.shape(x_input))
    pam_layer = PAM2()
    cam_layer = EcaBlockt2()
    pam_output = pam_layer(x_input)
    cam_output = cam_layer(x_input)
    output_summed = tf.add(pam_output, cam_output)
    output_final = Conv2D(filters=in_dim, kernel_size=(3, 3), padding='same')(
        output_summed)
    return output_final


def dual_attentiont3(x_input, in_dim):
    pam_layer = PAM3()
    cam_layer = EcaBlockt3()
    pam_output = pam_layer(x_input)
    cam_output = cam_layer(x_input)
    output_summed = tf.add(pam_output, cam_output)
    output_final = Conv2D(filters=in_dim, kernel_size=(3, 3), padding='same')(
        output_summed)
    return output_final


def dual_attentiont4(x_input, in_dim):
    pam_layer = PAM4()
    cam_layer = EcaBlockt4()
    pam_output = pam_layer(x_input)
    cam_output = cam_layer(x_input)
    output_summed = tf.add(pam_output, cam_output)
    output_final = Conv2D(filters=in_dim, kernel_size=(3, 3), padding='same')(
        output_summed)
    return output_final
