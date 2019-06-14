import tensorflow as tf
import time
import numpy as np
import os
import scipy.misc
from timeit import default_timer as timer
import scipy.misc
import os
import numpy as np

warm_up = 100
repeat = 100


def test():
    hr_pics_path = '../dataset/Set5/Set5/HR'
    lr_pics_path = '../dataset/Set5/Set5/LR'
    hr_pics = os.listdir(hr_pics_path)
    hr_pics = sorted(hr_pics)
    lr_pics = os.listdir(lr_pics_path)
    lr_pics = sorted(lr_pics)

    psnrs = []

    for i in range(len(hr_pics)):
        hr = scipy.misc.imread(os.path.join(hr_pics_path, hr_pics[i]))
        hr_y = hr[:, :, 0] * 0.299 + hr[:, :, 1] * 0.587 + hr[:, :, 2] * 0.114
        x, y, z = hr.shape
        # print(hr.shape)
        lr = scipy.misc.imread(os.path.join(lr_pics_path, lr_pics[i]))
        print(lr.shape)
        lr_bic = scipy.misc.imresize(lr, (x, y), 'bicubic')
        print(lr_bic.shape)
        lr_bic_y = lr_bic[:, :, 0] * 0.299 + lr_bic[:, :, 1] * 0.587 + lr_bic[:, :, 2] * 0.114
        mse = np.mean(np.square(hr_y - lr_bic_y))
        psnr = 10 * np.log10(255*255/mse)
        psnrs += [psnr]

    print(np.mean(psnrs))


def test_pb_model_speed(file_path, net_input_name, net_output_name, input_data):
    with tf.Graph().as_default() as graph:
        sess = tf.Session()

        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def)

        # obtain input & output nodes and then test the model
        net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
        net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')

        for i in range(warm_up + repeat):
            if i == warm_up:
                time_begin = timer()
            output_data = sess.run(net_output, feed_dict={net_input: input_data})

        time_cost = timer() - time_begin

        print(time_cost / repeat)
        return time_cost / repeat


def test_pb_model(file_path, net_input_name, net_output_name, input_data, label_data):
    with tf.Graph().as_default() as graph:
        sess = tf.Session()

        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def)

        # obtain input & output nodes and then test the model
        net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
        net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')

        output_data = sess.run(net_output, feed_dict={net_input: input_data})
        mse = np.mean(np.square(output_data - label_data))
        psnr = 255 * 255 / mse
        psnr = np.log10(psnr) * 10
        print(psnr)
        # print(output_data[0][0])
        # print(label_data[0][0])


def test_tflite_model(file_path, net_input_name, net_output_name, input_data, label_data):

  # restore the model and allocate tensors
  interpreter = tf.contrib.lite.Interpreter(model_path=file_path)
  interpreter.allocate_tensors()

  # get input & output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print('input details: {}'.format(input_details))
  print('output details: {}'.format(output_details))

  # test the model with given inputs
  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  net_output_data = interpreter.get_tensor(output_details[0]['index'])
  print('outputs from the *.tflite model: {}'.format(net_output_data))


def preprocess(pic_file, input_size, output_size):
    images, labels = [], []
    pic = scipy.misc.imread(pic_file)
    pic = pic[:, :, 0:3]
    x, y, z = pic.shape
    rows = x // output_size
    cols = y // output_size

    for r in range(rows):
        for c in range(cols):
            patch = pic[r * output_size: (r+1) * output_size, c * output_size: (c+1) * output_size, :]
            patch_low = scipy.misc.imresize(patch, (input_size, input_size))
            images += [patch_low]
            labels += [patch]

    # print(images[0])
    return images, labels


if __name__ == '__main__':
    test()
    # file_dir = 'test/test'
    # pic_dir = '../dataset/my-dataset/images'
    #
    # input_size = 48
    # scale = 2
    # output_size = scale * input_size
    #
    # net_input_name = 'net_input'
    # net_output_name = 'net_output'
    # # input_data = np.zeros(tuple([1] + [48, 48, 3]), dtype=np.float32)
    #
    # pic_file = os.path.join(pic_dir, 'test.jpg')
    # input_data, label_data = preprocess(pic_file, input_size, output_size)
    # input_data = np.array(input_data, np.float32)
    # label_data = np.array(label_data, np.float32)
    #
    # file_path = os.path.join(file_dir, 'model_original.pb')
    # print("original")
    # t_o = test_pb_model_speed(file_path, net_input_name, net_output_name, input_data)
    # test_pb_model(file_path, net_input_name, net_output_name, input_data, label_data)
    #
    # file_path = os.path.join(file_dir, 'model_transformed.pb')
    # print("transformed")
    # t_t = test_pb_model_speed(file_path, net_input_name, net_output_name, input_data)
    # test_pb_model(file_path, net_input_name, net_output_name, input_data, label_data)
    #
    # print(t_t - t_o)

