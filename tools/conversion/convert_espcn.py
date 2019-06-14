import tensorflow as tf
import numpy as np
import os
import re
import scipy.misc
import scipy.ndimage
import timeit

image_low = 48
scale = 2

def test():
    ckpt_dir = '../../models_dcp_eval/espcnp-2-48-dcp00'
    meta_file = ckpt_dir + '/model.ckpt.meta'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph = tf.get_default_graph()

    ops = graph.get_operations()
    trainable = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # pruned_model/conv2d_transpose/kernel/read:0
    # pruned_model/conv0/kernel/read:0
    # pruned_model/alpha0:0

    # for op in ops:
    #     # conv2d_pattern = re.compile(r'/Conv2D$')
    #     # if re.search(conv2d_pattern, op.name) is not None:
    #     if True:
    #         print(op.name)
    #         print("inputs")
    #         for input in op.inputs:
    #             print(input)
    #         print("outputs")
    #         for output in op.outputs:
    #             print(output)
    #         print()
    #
    for v in trainable:
        print(v.name)



    # var = graph.get_tensor_by_name('quant_model/conv0/kernel:0')
    # print(sess.run(var).dtype)

    # kernel = graph.get_tensor_by_name('pruned_model/alpha5:0')
    # print(sess.run(kernel))


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



def get_vars(ckpt_dir, scope):
    # ckpt_dir = '../../test/fsrcnn'
    meta_file = ckpt_dir + '/model.ckpt.meta'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph = tf.get_default_graph()

    res = {}
    # conv0
    name = scope + '/conv0/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'conv0_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv0_bias'
    res[new_name] = sess.run(var)

    # conv1
    name = scope + '/conv1/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'conv1_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv1_bias'
    res[new_name] = sess.run(var)

    # upsample
    name = scope + '/conv2d/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'conv2d_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv2d_bias'
    res[new_name] = sess.run(var)


    sess.close()
    return res


def get_vars_dcp(ckpt_dir, scope):
    # ckpt_dir = '../../test/fsrcnn-dcp50'
    meta_file = ckpt_dir + '/model.ckpt.meta'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph = tf.get_default_graph()

    res = {}
    # conv0
    name = scope + '/conv0/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name  ='conv0_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv0_bias'
    res[new_name] = sess.run(var)

    #conv1
    name = scope + '/conv1/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'conv1_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv1_bias'
    res[new_name] = sess.run(var)

    # upsample
    name = scope + '/conv2d/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'conv2d_kernel'
    res[new_name] = sess.run(var)

    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'conv2d_bias'
    res[new_name] = sess.run(var)

    sess.close()

    return res


def process_dcp_vars(kernels):
    NNZR = []

    name = 'conv0_kernel'
    kernel = kernels[name]
    nnzr = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
    NNZR.append(nnzr)
    name = 'conv1_kernel'
    kernel = kernels[name]
    nnzr = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
    NNZR.append(nnzr)
    name = 'conv2d_kernel'
    kernel = kernels[name]
    nnzr = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
    NNZR.append(nnzr)

    nnzr = NNZR[1]
    name = 'conv1_kernel'
    kernels[name] = kernels[name][:, :, nnzr, :]
    nnzr = NNZR[2]
    name = 'conv2d_kernel'
    kernels[name] = kernels[name][:, :, nnzr, :]

    nnzr_next = NNZR[1]
    name = 'conv0_kernel'
    kernels[name] = kernels[name][:, :, :, nnzr_next]
    name = 'conv0_bias'
    kernels[name] = kernels[name][nnzr_next]

    nnzr_next = NNZR[2]
    name = 'conv1_kernel'
    kernels[name] = kernels[name][:, :, :, nnzr_next]
    name = 'conv1_bias'
    kernels[name] = kernels[name][nnzr_next]


    for name in kernels.keys():
        print(kernels[name].shape)

    return kernels


def prelu(x, alpha):
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg


def model(kernels):

    LR = image_low
    HR = LR * scale

    net_inputs = tf.placeholder(tf.float32, shape=[None, LR, LR, 3], name='net_input')
    input_max = 255.0
    input_mean = 0

    net_inputs = (net_inputs - input_mean) / input_max

    # start
    inputs = tf.nn.conv2d(net_inputs, kernels['conv0_kernel'], [1, 1, 1, 1], 'SAME', name='conv0')
    inputs += kernels['conv0_bias']
    inputs = tf.nn.tanh(inputs)

    inputs = tf.nn.conv2d(inputs, kernels['conv1_kernel'], [1, 1, 1, 1], 'SAME', name='conv1')
    inputs += kernels['conv1_bias']
    inputs = tf.nn.tanh(inputs)

    inputs = tf.nn.conv2d(inputs, kernels['conv2d_kernel'], [1, 1, 1, 1], 'SAME', name='upsample')
    inputs += kernels['conv2d_bias']
    inputs = PS(inputs, scale, True)

    # inputs = tf.nn.sigmoid(inputs)

    inputs = inputs * input_max + input_mean
    inputs = tf.clip_by_value(inputs, 0.0, 255.0, name='net_output')
    return inputs


def test_psnr(Images, Labels, sess, graph, outputs):
    inputs = graph.get_tensor_by_name('net_input:0')

    # Images = np.array(Images, np.float32)
    # Labels = np.array(Labels, np.float32)
    psnrs = []
    bic_psnrs = []
    for i in range(len(Images)):
        images = Images[i]
        labels = Labels[i]
        res = sess.run([outputs], feed_dict={inputs: images})

        res_y = res[0]
        res_y = res_y[:, :, :, 0] * 0.299 + res_y[:, :, :, 1] * 0.587 + res_y[:, :, :, 2] * 0.114
        # res_y = np.expand_dims(res_y, 3)
        labels_y = np.array(labels, np.float32)
        labels_y = labels_y[:, :, :, 0] * 0.299 + labels_y[:, :, :, 1] * 0.587 + labels_y[:, :, :, 2] * 0.114
        # labels_y = np.expand_dims(labels_y, 3)
        # print((res_y - labels_y)[0][0])
        mse = np.mean(np.square(res_y - labels_y))
        # print(mse)
        psnr = np.log10(255 * 255 / mse) * 10
        psnrs += [psnr]
        print(psnr)
        # bics = []
        # for patch in images:
        #     bic = scipy.misc.imresize(patch, [54, 54])
        #     bics += [bic]
        #
        # mse = np.mean(np.square(bics - labels)) + 1
        # psnr = np.log10(255 * 255 / mse) * 10
        # bics += [psnr]

    print()
    print(np.mean(psnrs))
    # print(np.mean(bic_psnrs))
    return psnrs


def test_speed(Images, Labels, sess, graph, outputs, warmup, repeat):
    inputs = graph.get_tensor_by_name('net_input:0')

    # Images = np.array(Images, np.float32)
    # Labels = np.array(Labels, np.float32)
    images = Images[0]

    for i in range(warmup + repeat):
        if i == warmup:
            time_begin = timeit.default_timer()
        res = sess.run([outputs], feed_dict={inputs: images})
    time_cost = timeit.default_timer() - time_begin
    ms = time_cost / repeat / len(images) * 1000
    print('time: %.4f ms' % ms)


def preprocess(pic_file, input_size, output_size):
    images, labels = [], []
    mse_bicubic = []
    pic = scipy.misc.imread(pic_file)

    pic = pic[:, :, 0:3]

    x, y, z = pic.shape
    # print([x, y])
    rows = x // output_size
    cols = y // output_size

    # pic_low = scipy.misc.imresize(pic, )

    for r in range(rows):
        for c in range(cols):
            patch = pic[r * output_size: (r+1) * output_size, c * output_size: (c+1) * output_size, :]
            patch_low = scipy.misc.imresize(patch, (input_size, input_size), 'bicubic')

            patch_bic = scipy.misc.imresize(patch_low, (output_size, output_size), 'bicubic')
            patch_y = np.array(patch)
            patch_y = patch[:, :, 0] * 0.299 + patch[:, :, 1] * 0.587 + patch[:, :, 2] * 0.114
            patch_bic_y = np.array(patch_bic)
            patch_bic_y = patch_bic[:, :, 0] * 0.299 + patch_bic[:, :, 1] * 0.587 + patch_bic[:, :, 2] * 0.114
            mse = np.mean(np.square(patch_y - patch_bic_y))
            mse_bicubic += [mse]
            images += [patch_low]
            labels += [patch]
    mse_mean = np.mean(mse_bicubic)
    psnr = 10 * np.log10(255 * 255 / mse_mean)
    print(psnr)
    # print(images[0])
    return images, labels, psnr


def get_kernels(model_type, ckpt_dir):
    if model_type == 'original':
        return get_vars(ckpt_dir, 'model')
    if model_type == 'cp':
        kernels = get_vars_dcp(ckpt_dir, 'pruned_model0')
        return process_dcp_vars(kernels)


if __name__ == '__main__':
    # test()

    model_type = 'cp'
    ckpt_dir = '../../models_dcp_eval'
    # ckpt_dir = '../../models_eval'
    kernels = get_kernels(model_type, ckpt_dir)

    sess = tf.Session()

    graph = tf.get_default_graph()
    # saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    Images, Labels = [], []
    # test_pic_dir = '../../../dataset/BSD300/BSDS300/images/test'
    # test_pic_dir = '../../../dataset/BSD500/BSR/BSDS500/data/images/test'
    test_pic_dir = '../../../dataset/DIV2K_valid_HR'
    # test_pic_dir = '../../../dataset/Set5/Set5/HR'
    # test_pic_dir = '../../../dataset/Set14/HR'
    test_pics = os.listdir(test_pic_dir)
    count = 0
    psnr_bic = []
    for test_pic in test_pics:
        test_pic_full = os.path.join(test_pic_dir, test_pic)
        print(test_pic)
        images, labels, psnr = preprocess(test_pic_full, image_low, image_low * scale)
        psnr_bic += [psnr]
        Images.append(images)
        Labels += [labels]
        count += 1
    print()
    print(np.mean(psnr_bic))
    print()
    outputs = model(kernels)
    test_psnr(Images, Labels, sess, graph, outputs)

    print('bicubic: {}'.format(np.mean(psnr_bic)))

    test_speed(Images, Labels, sess, graph, outputs, 50, 10)
    #
    # graph_def = graph.as_graph_def()
    # graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['net_output'])
    # with tf.gfile.GFile('model_transformed.pb', mode='wb') as f:
    #     f.write(graph_def.SerializeToString())
    #
    # saver.save(sess, 'convert_test/')





