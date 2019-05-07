import os
import tensorflow as tf
import scipy.misc
from PIL import Image

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 1, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 235809, '# of samples for training 235809')
tf.app.flags.DEFINE_integer('nb_smpls_val', 1000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 30051, '# of samples for evaluation 30051')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 64, 'batch size for evaluation')

tf.app.flags.DEFINE_boolean('factory_mode', False, 'output an HR-image')
tf.app.flags.DEFINE_string('image_name', '', 'image name')
tf.app.flags.DEFINE_integer('input_size', 48, 'input patch size')
tf.app.flags.DEFINE_integer('sr_scale', 2, 'super resolution scale')

COLOR_CHANNEL = 3
image_size = 96
scale = 2
image_low = 48

def make_dataset():
    output_test_dir = os.path.join(FLAGS.data_dir_local, 'images/test.tfrecords')
    writer_test = tf.python_io.TFRecordWriter(output_test_dir)
    tmp_image = scipy.misc.imread(FLAGS.data_dir_local + "/images/" + FLAGS.image_name)

    global coordy
    global coordx
    x, y, z = tmp_image.shape
    tmp_image = tmp_image[:, :, 0:3]

    # print([x, y])
    coordx = x // image_low
    coordy = y // image_low
    for i in range(coordx):

        for j in range(coordy):
            tmp = tmp_image[i * image_low: (i + 1) * image_low, j * image_low: (j + 1) * image_low]
            tmp_high = scipy.misc.imresize(tmp, (image_size, image_size))
            tmp = tmp.tobytes()
            tmp_high = tmp_high.tobytes()

            features = tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tmp_high]))
            })
            example = tf.train.Example(features=features)

            writer_test.write(example.SerializeToString())

    writer_test.close()


def parse_fn(example_serialized, is_train):
    feature_map = {
        'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'label': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    image = features['image']
    label = features['label']

    image = tf.decode_raw(image, tf.uint8)
    label = tf.decode_raw(label, tf.uint8)

    # image = tf.image.decode_jpeg(image, COLOR_CHANNEL)
    # label = tf.image.decode_jpeg(label, COLOR_CHANNEL)

    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image = tf.reshape(image, [image_low, image_low, COLOR_CHANNEL])
    label = tf.reshape(label, [image_size, image_size, COLOR_CHANNEL])

    # print(image.get_shape().as_list())
    # print(image.get_shape().as_list())
    # print(image.get_shape().as_list())

    return image, label


class EdsrDataset(AbstractDataset):
    def __init__(self, is_train):
        super(EdsrDataset, self).__init__(is_train)

        if FLAGS.data_disk == 'local':
            assert FLAGS.data_dir_local is not None, '<FLAGS.data_dir_local> must not be None'
            data_dir = FLAGS.data_dir_local
        elif FLAGS.data_disk == 'hdfs':
            assert FLAGS.data_hdfs_host is not None and FLAGS.data_dir_hdfs is not None, \
                'both <FLAGS.data_hdfs_host> and <FLAGS.data_dir_hdfs> must not be None'
            data_dir = FLAGS.data_hdfs_host + FLAGS.data_dir_hdfs
        else:
            raise ValueError('unrecognized data disk: ' + FLAGS.data_disk)

        if FLAGS.factory_mode:
            make_dataset()

        # configure file patterns & function handlers
        if is_train:
            self.file_pattern = os.path.join(data_dir, 'train*')
            self.batch_size = FLAGS.batch_size
        else:
            self.file_pattern = os.path.join(data_dir, 'test*')
            if FLAGS.factory_mode:
                self.file_pattern = os.path.join(data_dir, 'images/test.tfrecords')
            self.batch_size = FLAGS.batch_size_eval
        self.dataset_fn = tf.data.TFRecordDataset
        self.parse_fn = lambda x: parse_fn(x, is_train=is_train)



