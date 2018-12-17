import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.abstract_model_helper import AbstractModelHelper

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('lrn_rate_init', 1e-2, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 5e-4, 'weight decaying loss\'s coefficient')


def resBlock(x, index, data_format, channels=256, kernel_size=[3, 3], scale=0.1):
    tmp = tf.layers.conv2d(x, channels, kernel_size, activation_fn=None, data_format=data_format,
                           name="conv"+index+"-1")
    tmp = tf.nn.relu(tmp)
    tmp = tf.layers.conv2d(tmp, channels, kernel_size, activation_fn=None, data_format=data_format,
                           name="conv"+index+"-2")
    tmp *= scale
    return x + tmp


def upsample(x, scale=2, features=64, activation=tf.nn.relu):
    assert scale in [2, 3, 4]
    x = slim.conv2d(x, features, [3, 3], activation_fn=activation)
    if scale == 2:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
        x = PS(x, 2, color=True)
    elif scale == 3:
        ps_features = 3 * (scale ** 2)
        x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
        # x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
        x = PS(x, 3, color=True)
    elif scale == 4:
        ps_features = 3 * (2 ** 2)
        for i in range(2):
            x = slim.conv2d(x, ps_features, [3, 3], activation_fn=activation)
            # x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = PS(x, 2, color=True)
    return x


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
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
    features = 256
    layer_num = 32
    scaling_factor = 0.1
    scale = 2

    inputs = inputs - tf.reduce_mean(inputs)
    inputs = tf.layers.conv2d(inputs, features, [3, 3], data_format=data_format, name="conv0")
    conv0 = inputs

    for i in range(layer_num):
        inputs = resBlock(inputs, i, data_format, features, [3, 3], scaling_factor)

    inputs = tf.layers.conv2d(inputs, features, [3, 3], data_format=data_format, name="conv-1")
    inputs += conv0

    inputs = upsample(inputs, scale, features, None)

    return inputs


class ModelHelper(AbstractModelHelper):
    def __init__(self):
        super(ModelHelper, self).__init__()

    def forward_train(self, inputs, data_format='channels_last'):
        return forward_fn(inputs, data_format)

    def forward_eval(self, inputs, data_format='channels_last'):
        return forward_fn(inputs, data_format)

    def calc_loss(self, labels, outputs, trainable_vars):
        loss = tf.reduce_mean(tf.losses.absolute_difference(labels, outputs))
        loss += FLAGS.loss_w_dcy * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])

        mse = tf.reduce_mean(tf.squared_difference(labels, outputs))
        PSNR = tf.constant(255 ** 2, dtype=tf.float32) / mse
        PSNR = tf.constant(10, dtype=tf.float32) * log10(PSNR)
        accuracy = PSNR
        metrics = {'accuracy': accuracy}

        return loss, metrics

    @property
    def model_name(self):
        """Model's name."""

        return 'edsr'

    @property
    def dataset_name(self):
        """Dataset's name."""

        return 'myDataset'

