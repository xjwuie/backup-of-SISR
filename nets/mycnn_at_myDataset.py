import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
from PIL import Image
import os

from nets.abstract_model_helper import AbstractModelHelper
from datasets.edsr_dataset import EdsrDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('lrn_rate_init', 0.0002, 'initial learning rate  l1=0.0002')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 5e-6, 'weight decaying loss\'s coefficient')


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
    a = b = FLAGS.input_size
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


def prelu(x, i):
    alphas = tf.get_variable('alpha{}'.format(i), x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    # neg = tf.nn.relu(-x)
    # neg = alphas * (-neg)

    return pos + neg


def conv(x, index, data_format, channels, kernel_size=[3, 3]):
    x = tf.layers.conv2d(x, channels, kernel_size, data_format=data_format,
                         padding='SAME', name='conv'+str(index))
    # x = prelu(x, index)
    # x = tf.nn.relu(x)

    return x


def forward_fn(inputs, data_format):
    d = 64
    s = 16
    m = 4
    scale = FLAGS.sr_scale

    inputs_mean = 127.0
    inputs_max = 255.0
    # inputs = inputs - inputs_mean
    inputs = inputs / inputs_max

    inputs = tf.layers.conv2d(inputs, 3, [1, 1], data_format=data_format, padding='SAME', name='gamma0', use_bias=False)

    inputs = tf.layers.conv2d(inputs, d, [5, 5], data_format=data_format, padding='SAME', name='conv0')
    inputs = prelu(inputs, 0)
    # inputs = tf.nn.relu(inputs)

    inputs = tf.layers.conv2d(inputs, s, [1, 1], data_format=data_format, padding='SAME', name='conv1')
    inputs = prelu(inputs, 1)
    # inputs = tf.nn.relu(inputs)

    for i in range(2, 2+m):
        inputs = conv(inputs, i, data_format, s, [3, 3])
        # inputs = tf.nn.relu(inputs)
        inputs = prelu(inputs, i)

    inputs = tf.layers.conv2d(inputs, d, [1, 1], data_format=data_format, padding='SAME', name='conv'+str(2+m))
    inputs = prelu(inputs, 2+m)
    # inputs = tf.nn.relu(inputs)
    # inputs = tf.nn.conv2d_transpose(inputs, [3, 3, 3, d],
    #                                 [inputs.get_shape()[0], HR_size, HR_size, 3],
    #                                 strides=[1, 2, 2, 1],
    #                                 data_format=data_format)
    inputs = tf.layers.conv2d_transpose(inputs, 3, [9, 9], (scale, scale),
                                        padding='SAME', data_format=data_format)

    inputs = tf.layers.conv2d(inputs, 3, [1, 1], data_format=data_format, padding='SAME', name='gamma1', use_bias=False)

    inputs = inputs * inputs_max
    # inputs = tf.clip_by_value(inputs, 0.0, 255.0)
    return inputs


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
        # loss = tf.reduce_mean(tf.nn.l2_loss(labels - outputs))
        loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])

        labels_y = labels[:, :, 0] * 0.299 + labels[:, :, 1] * 0.587 + labels[:, :, 2] * 0.114
        outputs_y = outputs[:, :, 0] * 0.299 + outputs[:, :, 1] * 0.587 + outputs[:, :, 2] * 0.114
        mse_y = tf.reduce_mean(tf.squared_difference(labels_y, outputs_y))
        mse = tf.reduce_mean(tf.squared_difference(labels, outputs), axis=[1, 2, 3])
        MSE = tf.reduce_mean(mse)
        # print("mse shape: ", mse.shape)
        # PSNR = tf.constant(255 ** 2, dtype=tf.float32, shape=mse.shape)
        PSNR = tf.divide(255.0 ** 2, MSE)
        PSNR_Y = tf.divide(255.0 ** 2, mse_y)

        PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)
        PSNR_Y = tf.constant(10, dtype=tf.float32) * log10(PSNR_Y)

        # PSNR = tf.reduce_mean(PSNR)

        SSIM = tf.image.ssim(outputs, labels, 255)
        SSIM = tf.reduce_mean(SSIM)

        metrics = {'PSNR': PSNR, 'PSNR_Y': PSNR_Y, 'MSE': MSE, 'SSIM': SSIM}

        return loss, metrics

    @property
    def model_name(self):
        """Model's name."""

        return 'mycnn'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'myDataset'

