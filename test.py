import tensorflow as tf
import time
import numpy as np
import os
import scipy.misc
from timeit import default_timer as timer


warm_up = 100
repeat = 100


def test():
    print("2019-05-05 18:51:18.402914: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA\n2019-05-05 18:51:18.458970: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero\n2019-05-05 18:51:18.459442: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: \nname: GeForce GTX 980M major: 5 minor: 2 memoryClockRate(GHz): 1.1265\npciBusID: 0000:01:00.0\ntotalMemory: 3.94GiB freeMemory: 3.36GiB\n2019-05-05 18:51:18.459466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0\n2019-05-05 18:51:18.679390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:\n2019-05-05 18:51:18.679433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 \n2019-05-05 18:51:18.679439: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N \n2019-05-05 18:51:18.679594: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3073 MB memory) -> physical GPU (device: 0, name: GeForce GTX 980M, pci bus id: 0000:01:00.0, compute capability: 5.2)\n2019-05-05 18:51:18.688765: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before Removing unused ops: 114 operators, 186 arrays (0 quantized)\n2019-05-05 18:51:18.689694: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before general graph transformations: 114 operators, 186 arrays (0 quantized)\n2019-05-05 18:51:18.690848: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 1: 24 operators, 48 arrays (1 quantized)\n2019-05-05 18:51:18.690976: W tensorflow/contrib/lite/toco/tooling_util.cc:1685] Dropping MinMax information in array quant_model/gamma0/weights_quant/FakeQuantWithMinMaxVars. Expect inaccuracy in quantized inference.\n2019-05-05 18:51:18.691106: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After general graph transformations pass 2: 23 operators, 46 arrays (1 quantized)\n2019-05-05 18:51:18.691316: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before pre-quantization graph transformations: 23 operators, 46 arrays (1 quantized)\n2019-05-05 18:51:18.691409: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After pre-quantization graph transformations pass 1: 14 operators, 37 arrays (1 quantized)\n2019-05-05 18:51:18.691504: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before default min-max range propagation graph transformations: 14 operators, 37 arrays (1 quantized)\n2019-05-05 18:51:18.691579: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] After default min-max range propagation graph transformations pass 1: 14 operators, 37 arrays (1 quantized)\n2019-05-05 18:51:18.691674: I tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.cc:39] Before quantization graph transformations: 14 operators, 37 arrays (1 quantized)\n2019-05-05 18:51:18.691695: W tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:100] Constant array quant_model/gamma0/weights_quant/FakeQuantWithMinMaxVars lacks MinMax information. To make up for that, we will now compute the MinMax from actual array elements. That will result in quantization parameters that probably do not match whichever arithmetic was used during training, and thus will probably be a cause of poor inference accuracy.\n2019-05-05 18:51:18.691992: F tensorflow/contrib/lite/toco/graph_transformations/quantize.cc:474] Unimplemented: this graph contains an operator of type TransposeConv for which the quantized form is not yet implemented. Sorry, and patches welcome (that's a relatively fun patch to write, mostly providing the actual quantized arithmetic code for this op).\nAborted (core dumped)\n")
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

