import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from PIL import Image
import os

from nets.abstract_model_helper import AbstractModelHelper
from datasets.edsr_dataset import EdsrDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('lrn_rate_init', 0.0002, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 5e-6, 'weight decaying loss\'s coefficient')

image_low = 48



def resBlock(x, index, data_format, channels=256, kernel_size=[3, 3], scale=0.1):
    tmp = tf.layers.conv2d(x, channels, kernel_size, data_format=data_format, padding='SAME',
                           name="conv"+str(index)+"-1")
    # tmp = slim.conv2d(x, channels, kernel_size)
    tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv2d(tmp, channels, kernel_size, data_format=data_format, padding='SAME',
                           name="conv"+str(index)+"-2")
    # tmp = slim.conv2d(tmp, channels, kernel_size)
    tmp *= scale
    return x + tmp


def upsample(x, scale=2, features=64, activation=None):
    assert scale in [2, 3, 4]
    # x = tf.layers.conv2d(x, features, [3, 3], padding='SAME')
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = tf.layers.conv2d(x, ps_features, [3, 3], padding='SAME')
        # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = tf.layers.conv2d(x, ps_features, [3, 3], padding='SAME')
        # x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = tf.layers.conv2d(x, ps_features, [3, 3], padding='SAME')
            # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x, 2, color=True)
    return x


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    dims = I.get_shape().as_list()
    a = b = image_low
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, shape=[-1, a, b, r, r])
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def forward_fn(inputs, data_format):
    features0 = 64
    filter0 = 5
    features1 = 32
    filter1 = 3
    filter3 = 3

    scale = 2

    inputs_mean = 127.0
    inputs = inputs - inputs_mean
    conv0 = tf.layers.conv2d(inputs, features0, [filter0, filter0],
                             data_format=data_format, padding='SAME', name="conv0")

    conv1 = tf.layers.conv2d(conv0, features1, [filter1, filter1],
                             data_format=data_format, padding='SAME', name="conv1")

    outputs = upsample(conv1, scale, features1, None)
    # print(inputs.get_shape().as_list())
    outputs = tf.clip_by_value(outputs + inputs_mean, 0.0, 255.0)

    return outputs


class ModelHelper(AbstractModelHelper):
    def __init__(self):
        super(ModelHelper, self).__init__()

        self.dataset_train = EdsrDataset(is_train=True)
        self.dataset_eval = EdsrDataset(is_train=False)

        self.out_idx = 0

    def build_dataset_train(self, enbl_trn_val_split=False):
        """Build the data subset for training, usually with data augmentation."""

        return self.dataset_train.build(enbl_trn_val_split)

    def build_dataset_eval(self):
        """Build the data subset for evaluation, usually without data augmentation."""

        return self.dataset_eval.build()

    def forward_train(self, inputs, data_format='channels_last'):
        return forward_fn(inputs, data_format)

    def forward_eval(self, inputs, data_format='channels_last'):
        return forward_fn(inputs, data_format)

    def calc_loss(self, labels, outputs, trainable_vars):
        # t = time.time()
        # outputs = outputs - tf.reduce_mean(outputs)
        # labels = labels - tf.reduce_mean(labels)
        # outputs = outputs - tf.reduce_mean(outputs)
        loss = tf.reduce_mean(tf.losses.absolute_difference(labels, outputs))
        loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])

        mse = tf.reduce_mean(tf.squared_difference(labels, outputs), axis=[1, 2, 3])
        # print("mse shape: ", mse.shape)
        # PSNR = tf.constant(255 ** 2, dtype=tf.float32, shape=mse.shape)
        PSNR = tf.divide(255 ** 2, mse)

        PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)
        PSNR = tf.reduce_mean(PSNR)
        accuracy = PSNR
        metrics = {'PSNR': accuracy}

        return loss, metrics

    @property
    def model_name(self):
        """Model's name."""

        return 'espcn'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'myDataset'

