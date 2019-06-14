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


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def prelu(x, i):
    alphas = tf.get_variable('alpha{}'.format(i), x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5

    return pos + neg


def conv(x, index, data_format, channels, kernel_size=[3, 3]):
    x = tf.layers.conv2d(x, channels, kernel_size, data_format=data_format,
                         padding='SAME', name='conv'+str(index))
    x = prelu(x, index)

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

    inputs = prelu(
        tf.layers.conv2d(inputs, d, [5, 5], data_format=data_format, padding='SAME', name='conv0'),
        0
    )
    inputs = prelu(
        tf.layers.conv2d(inputs, s, [1, 1], data_format=data_format, padding='SAME', name='conv1'),
        1
    )

    for i in range(2, 2+m):
        inputs = conv(inputs, i, data_format, s, [3, 3])

    inputs = prelu(
        tf.layers.conv2d(inputs, d, [1, 1], data_format=data_format, padding='SAME', name='conv'+str(2+m)),
        2+m
    )
    # inputs = tf.nn.conv2d_transpose(inputs, [3, 3, 3, d],
    #                                 [inputs.get_shape()[0], HR_size, HR_size, 3],
    #                                 strides=[1, 2, 2, 1],
    #                                 data_format=data_format)
    inputs = tf.layers.conv2d_transpose(inputs, 3, [9, 9], (scale, scale),
                                        padding='SAME', data_format=data_format)

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

        return 'fsrcnn'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'myDataset'

