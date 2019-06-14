import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import scipy.misc
from PIL import Image
import os

from nets.abstract_model_helper import AbstractModelHelper
from datasets.edsr_dataset import EdsrDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('lrn_rate_init', 0.0002, 'initial learning rate  l1=0.0002')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 5e-4, 'weight decaying loss\'s coefficient')

image_size = 96
sr_scale = 2
image_low = 48


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
    # X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
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

    return pos + neg

def forward_fn(inputs, data_format):
    features0 = 64
    filter0 = 5
    features1 = 32
    filter1 = 3
    filter3 = 3

    scale = sr_scale

    inputs_mean = 127.0
    inputs_max = 255.0
    inputs = inputs
    # inputs = inputs - inputs_mean
    inputs = inputs / inputs_max
    conv0 = tf.layers.conv2d(inputs, features0, [filter0, filter0],
                             data_format=data_format, padding='SAME', name="conv0",
                             activation=None)
    conv0 = prelu(conv0, 0)

    conv1 = tf.layers.conv2d(conv0, features1, [filter1, filter1],
                             data_format=data_format, padding='SAME', name="conv1",
                             activation=None)

    conv1 = prelu(conv1, 1)

    outputs = upsample(conv1, scale, features1, None)
    # outputs = tf.nn.sigmoid(outputs)
    # print(inputs.get_shape().as_list())
    outputs = outputs * inputs_max
    # outputs = outputs + inputs_mean
    # outputs = tf.clip_by_value(outputs, 0.0, 255.0)

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
        # loss = tf.reduce_mean(tf.squared_difference(labels, outputs))
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

        SSIM = tf.image.ssim(outputs, labels, 255)
        SSIM = tf.reduce_mean(SSIM)

        # accuracy = PSNR
        # bilinear = labels
        # bilinear = tf.image.resize_bilinear(bilinear, (48, 48))
        # # print(bilinear.shape)
        # # print(bilinear.dtype)
        # bilinear = tf.image.resize_bilinear(bilinear, (96, 96))
        # bicubic = tf.image.resize_bicubic(bilinear, (96, 96))
        # # print(bilinear.shape)
        # # print(bilinear.dtype)
        # mse_bilinear = tf.reduce_mean(tf.squared_difference(bilinear, labels), axis=[1, 2, 3])
        # PSNR_bilinear = mse_bilinear + 1.0
        # PSNR_bilinear = tf.divide(255 ** 2, PSNR_bilinear)
        # PSNR_bilinear = tf.constant(10, dtype=tf.float32) * log10(PSNR_bilinear)
        # PSNR_bilinear = tf.reduce_mean(PSNR_bilinear)

        metrics = {'PSNR': PSNR, 'PSNR_Y': PSNR_Y, 'MSE': MSE, 'SSIM': SSIM}

        return loss, metrics

    @property
    def model_name(self):
        """Model's name."""

        return 'espcn'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'myDataset'

