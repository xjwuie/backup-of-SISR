import tensorflow as tf
import numpy as np
import os
import re
import scipy.misc
import timeit

image_low = 48
scale = 2
m = 4

def test():
    ckpt_dir = '../../models_dcp_eval'
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
    # #
    # print('\n\n\n')
    for v in trainable:
        print(v.name)



    # var = graph.get_tensor_by_name('quant_model/conv0/kernel:0')
    # print(sess.run(var).dtype)

    # kernel = graph.get_tensor_by_name('pruned_model/alpha5:0')
    # print(sess.run(kernel))


def get_vars(ckpt_dir, scope):
    # ckpt_dir = '../../test/fsrcnn'
    meta_file = ckpt_dir + '/model.ckpt.meta'

    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

    graph = tf.get_default_graph()

    res = {}

    for i in range(2):
        name = scope + '/gamma{}/kernel/read:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name  ='gamma{}_kernel'.format(i)
        res[new_name] = sess.run(var)

        name = name.replace('kernel', 'bias')
        var = graph.get_tensor_by_name(name)
        new_name = 'gamma{}_bias'.format(i)
        res[new_name] = sess.run(var)

    for i in range(m + 3):
        name = scope + '/conv{}/kernel/read:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name  ='conv{}_kernel'.format(i)
        res[new_name] = sess.run(var)

        name = name.replace('kernel', 'bias')
        var = graph.get_tensor_by_name(name)
        new_name = 'conv{}_bias'.format(i)
        res[new_name] = sess.run(var)

        name = scope + '/alpha{}:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name = 'alpha{}'.format(i)
        res[new_name] = sess.run(var)
    name = scope + '/conv2d_transpose/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'convT_kernel'
    res[new_name] = sess.run(var)
    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'convT_bias'
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

    for i in range(2):
        name = scope + '/gamma{}/kernel/read:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name  ='gamma{}_kernel'.format(i)
        res[new_name] = sess.run(var)
        print(res[new_name])

        name = name.replace('kernel', 'bias')
        var = graph.get_tensor_by_name(name)
        new_name = 'gamma{}_bias'.format(i)
        res[new_name] = sess.run(var)
        print(res[new_name])

    for i in range(m + 3):
        name = scope + '/conv{}/kernel/read:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name  ='conv{}_kernel'.format(i)
        res[new_name] = sess.run(var)

        name = name.replace('kernel', 'bias')
        var = graph.get_tensor_by_name(name)
        new_name = 'conv{}_bias'.format(i)
        res[new_name] = sess.run(var)

        name = scope + '/alpha{}:0'.format(i)
        var = graph.get_tensor_by_name(name)
        new_name = 'alpha{}'.format(i)
        res[new_name] = sess.run(var)
    name = scope + '/conv2d_transpose/kernel/read:0'
    var = graph.get_tensor_by_name(name)
    new_name = 'convT_kernel'
    res[new_name] = sess.run(var)
    name = name.replace('kernel', 'bias')
    var = graph.get_tensor_by_name(name)
    new_name = 'convT_bias'
    res[new_name] = sess.run(var)

    sess.close()

    return res


def process_dcp_vars(kernels):
    NNZR = []
    for i in range(m+3):
        name = 'conv{}_kernel'.format(i)
        kernel = kernels[name]
        # print(kernel.shape)
        nnzr = np.nonzero(np.sum(np.abs(kernel), axis=(0, 1, 3)))[0]
        NNZR.append(nnzr)

    # print()
    for i in range(1, m+3):
        nnzr = NNZR[i]
        name = 'conv{}_kernel'.format(i)
        kernels[name] = kernels[name][:, :, nnzr, :]

    for i in range(m+2):
        nnzr_next = NNZR[i + 1]
        name = 'conv{}_kernel'.format(i)
        kernels[name] = kernels[name][:, :, :, nnzr_next]

        name = 'conv{}_bias'.format(i)
        kernels[name] = kernels[name][nnzr_next]

        name = 'alpha{}'.format(i)
        kernels[name] = kernels[name][nnzr_next]

    for name in kernels.keys():
        print(kernels[name].shape)

    return kernels


def prelu(x, alpha):
    pos = tf.nn.relu(x)
    neg = alpha * (x - tf.abs(x)) * 0.5
    # print(x.shape)
    # neg = tf.nn.relu(-x)
    # print(neg.shape)
    # neg = alpha * (-neg)
    return pos + neg


def model(kernels):
    d = 64
    s = 16
    # m = 4

    LR = image_low
    HR = LR * scale

    inputs_max = 255.0

    net_inputs = tf.placeholder(tf.float32, shape=[None, LR, LR, 3], name='net_input')
    inputs = net_inputs / inputs_max

    inputs = tf.nn.conv2d(inputs, kernels['gamma0_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='gamma0')
    inputs += kernels['gamma0_bias']
    # start
    inputs = tf.nn.conv2d(inputs, kernels['conv0_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv0')
    inputs += kernels['conv0_bias']
    inputs = prelu(inputs, kernels['alpha0'])

    inputs = tf.nn.conv2d(inputs, kernels['conv1_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv1')
    inputs += kernels['conv1_bias']
    inputs = prelu(inputs, kernels['alpha1'])

    for i in range(2, 2+m):
        conv_idx = 'conv' + str(i)
        inputs = tf.nn.conv2d(inputs, kernels[conv_idx + '_kernel'], [1, 1, 1, 1],
                              'SAME', data_format='NHWC', name=conv_idx)
        inputs += kernels[conv_idx + '_bias']
        inputs = prelu(inputs, kernels['alpha' + str(i)])
    # middle
    # inputs = tf.nn.conv2d(inputs, kernels['conv2_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv2')
    # inputs += kernels['conv2_bias']
    # inputs = prelu(inputs, kernels['alpha2'])
    #
    # inputs = tf.nn.conv2d(inputs, kernels['conv3_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv3')
    # inputs += kernels['conv3_bias']
    # inputs = prelu(inputs, kernels['alpha3'])
    #
    # inputs = tf.nn.conv2d(inputs, kernels['conv4_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv4')
    # inputs += kernels['conv4_bias']
    # inputs = prelu(inputs, kernels['alpha4'])
    #
    # inputs = tf.nn.conv2d(inputs, kernels['conv5_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='conv5')
    # inputs += kernels['conv5_bias']
    # inputs = prelu(inputs, kernels['alpha5'])

    # end
    conv_idx = 'conv' + str(2 + m)
    inputs = tf.nn.conv2d(inputs, kernels[conv_idx + '_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name=conv_idx)
    inputs += kernels[conv_idx + '_bias']
    inputs = prelu(inputs, kernels['alpha' + str(2 + m)])

    inputs = tf.nn.conv2d_transpose(
        inputs, kernels['convT_kernel'], [tf.shape(net_inputs)[0], HR, HR, 3], [1, scale, scale, 1], data_format='NHWC', name='convT')
    inputs += kernels['convT_bias']

    inputs = tf.nn.conv2d(inputs, kernels['gamma1_kernel'], [1, 1, 1, 1], 'SAME', data_format='NHWC', name='gamma1')
    inputs += kernels['gamma1_bias']

    inputs = tf.clip_by_value(inputs * inputs_max, 0.0, 255.0, name='net_output')
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
    ms = time_cost / repeat * 1000
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

    pic_low = scipy.misc.imresize(pic, (x // scale, y // scale))
    # print(pic_low.shape)
    pic_bic = scipy.misc.imresize(pic_low, (x, y), 'bicubic')
    # print(pic_bic.shape)

    pic_y = pic[:, :, 0] * 0.299 + pic[:, :, 1] * 0.587 + pic[:, :, 2] * 0.114
    pic_bic_y = pic_bic[:, :, 0] * 0.299 + pic_bic[:, :, 1] * 0.587 + pic_bic[:, :, 2] * 0.114
    mse_pic_y = np.mean(np.square(pic_bic_y - pic_y))
    mse_pic = np.mean(np.square(pic_bic - pic))
    psnr_pic_y = np.log10(255 * 255 / mse_pic_y) * 10
    psnr_pic = np.log10(255 * 255 / mse_pic) * 10

    print(psnr_pic_y)

    for r in range(rows):
        for c in range(cols):
            patch = pic[r * output_size: (r+1) * output_size, c * output_size: (c+1) * output_size, :]
            patch_low = scipy.misc.imresize(patch, (input_size, input_size))

            patch_bic = scipy.misc.imresize(patch_low, (output_size, output_size))
            patch_y = patch[:, :, 0] * 0.299 + patch[:, :, 1] * 0.587 + patch[:, :, 2] * 0.114
            patch_bic_y = patch_bic[:, :, 0] * 0.299 + patch_bic[:, :, 1] * 0.587 + patch_bic[:, :, 2] * 0.114
            mse = np.mean(np.square(patch_y - patch_bic_y))
            mse_bicubic += [mse]
            images += [patch_low]
            labels += [patch]
    mse_mean = np.mean(mse_bicubic)
    psnr = 10 * np.log10(255 * 255 / mse_mean)
    # print(psnr)
    # print(images[0])
    return images, labels, psnr, psnr_pic_y


def get_kernels(model_type, ckpt_dir, model_scope, pruned_scope):
    if model_type == 'original':
        return get_vars(ckpt_dir, model_scope)
    if model_type == 'cp':
        kernels = get_vars_dcp(ckpt_dir, pruned_scope)
        return process_dcp_vars(kernels)

if __name__ == '__main__':
    # test()

    model_type = 'cp'
    ckpt_dir = '../../test/mycnn-2-48-dcp0-dcp0'
    # ckpt_dir = '../../models_eval'
    model_scope = 'model'
    pruned_scope = 'pruned_model_0'
    kernels = get_kernels(model_type, ckpt_dir, model_scope, pruned_scope)

    sess = tf.Session()

    graph = tf.get_default_graph()
    # saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    Images, Labels = [], []
    # test_pic_dir = '../../../dataset/BSD300/BSDS300/images/test'
    # test_pic_dir = '../../../dataset/BSD500/BSR/BSDS500/data/images/test'

    # gamma  32.36/134.97
    # m=3 32.33/132.31
    test_pic_dir = '../../../dataset/DIV2K_valid_HR'
    # test_pic_dir = '../../../dataset/Set5/Set5/HR'
    # test_pic_dir = '../../../dataset/Set14/HR'
    test_pics = os.listdir(test_pic_dir)
    test_pics = sorted(test_pics)
    # if len(test_pics) > 20:
    #     test_pics = test_pics[: 20]
    count = 0
    psnr_bic = []

    for test_pic in test_pics:
        test_pic_full = os.path.join(test_pic_dir, test_pic)
        print(test_pic)
        images, labels, psnr, psnr_pic = preprocess(test_pic_full, image_low, image_low * scale)
        psnr_bic += [psnr_pic]
        Images.append(images)
        Labels += [labels]
        count += 1
    print()
    print(np.mean(psnr_bic))
    print()
    outputs = model(kernels)
    test_psnr(Images, Labels, sess, graph, outputs)

    print('bicubic: {}'.format(np.mean(psnr_bic)))
    #
    test_speed(Images, Labels, sess, graph, outputs, 50, 50)

    # graph_def = graph.as_graph_def()
    # graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['net_output'])
    # with tf.gfile.GFile('model_transformed.pb', mode='wb') as f:
    #     f.write(graph_def.SerializeToString())
    #
    # saver.save(sess, 'convert_test/')





