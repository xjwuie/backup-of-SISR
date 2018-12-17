import os
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_smpls_train', 1281167, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 10000, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 50000, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 64, 'batch size for evaluation')

COLOR_CHANNEL = 3


def parse_fn(example_serialized, is_train):
    feature_map = {
        'image_lr': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image_hr': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    image = features['image_lr']
    label = features['image_hr']

    image = tf.image.decode_jpeg(image, COLOR_CHANNEL)
    label = tf.image.decode_jpeg(label, COLOR_CHANNEL)

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

        # configure file patterns & function handlers
        if is_train:
            self.file_pattern = os.path.join(data_dir, 'train*')
            self.batch_size = FLAGS.batch_size
        else:
            self.file_pattern = os.path.join(data_dir, 'test*')
            self.batch_size = FLAGS.batch_size_eval
        self.dataset_fn = tf.data.TFRecordDataset
        self.parse_fn = lambda x: parse_fn(x, is_train=is_train)



