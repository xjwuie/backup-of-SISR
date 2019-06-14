import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.ndimage
from PIL import Image
import os

# data_dir = "BSD300/BSDS300/images/train"
# data_dir = "BSD500/BSR/BSDS500/data/images/test"
data_dir = "DIV2K_train_HR"
# data_dir = "BSD/test"
# test_dir = "data/test"
output_data_dir = "train.tfrecords"
output_test_dir = "test.tfrecords"
read_data_dir = "read/"
image_size = 96
scale = 2
image_low = 48

write = True
read = False

if write:
    image_files = os.listdir(data_dir)
    # test_files = os.listdir(test_dir)

    train = []
    label = []
    test_image = []
    test_label = []

    writer_train = tf.python_io.TFRecordWriter(output_data_dir)
    writer_test = tf.python_io.TFRecordWriter(output_test_dir)
    count_train = 0
    count_test = 0
    for image_name in image_files:
        print(image_name)
        tmp_image = scipy.misc.imread(data_dir + "/" + image_name)

        x, y, z = tmp_image.shape
        tmp_image = tmp_image[:, :, 0:3]
        # tmp_image_low = scipy.misc.imresize(tmp_image, (x // scale, y // scale), 'bicubic')
        # print([x, y])
        coordx = x // image_size
        coordy = y // image_size
        for i in range(coordx):

            for j in range(coordy):
                tmp = tmp_image[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size]
                tmp_low = scipy.misc.imresize(tmp, (image_low, image_low), 'bicubic')
                # tmp_low = tmp_image_low[i * image_low: (i + 1) * image_low, j * image_low: (j + 1) * image_low]
                # tmp_low = scipy.ndimage.gaussian_filter(tmp_low, 2)
                tmp = tmp.tobytes()
                tmp_low = tmp_low.tobytes()

                features = tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_low])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp]))
                })
                example = tf.train.Example(features=features)
                if np.random.random() > -2:
                    writer_train.write(example.SerializeToString())
                    count_train += 1
                else:
                    count_test += 1
                    # print("test")
                    writer_test.write(example.SerializeToString())
    writer_train.close()
    writer_test.close()
    print(count_train)
    print(count_test)


read_num = 8
if read:
    filename_queue = tf.train.string_input_producer([output_test_dir])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })
    image_read = tf.decode_raw(features['image'], tf.uint8)
    label_read = tf.decode_raw(features['label'], tf.uint8)
    image_read = tf.reshape(image_read, [image_size//scale, image_size//scale, 3])
    label_read = tf.reshape(label_read, [image_size, image_size, 3])

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(read_num):
            example, label = sess.run([image_read, label_read])
            img = Image.fromarray(example, 'RGB')
            img.save(read_data_dir + str(i) + 'image.jpg')
            img = Image.fromarray(label, 'RGB')
            img.save(read_data_dir + str(i) + 'label.jpg')

        coord.request_stop()
        coord.join(threads)




