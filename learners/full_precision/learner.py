# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Full-precision learner (no model compression applied)."""

import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import scipy.misc

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from utils.lrn_rate_utils import setup_lrn_rate
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

from datasets import edsr_dataset

FLAGS = tf.app.flags.FLAGS

class FullPrecLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
    """Full-precision learner (no model compression applied)."""

    def __init__(self, sm_writer, model_helper, model_scope=None, enbl_dst=None):
        """Constructor function.

        Args:
        * sm_writer: TensorFlow's summary writer
        * model_helper: model helper with definitions of model & dataset
        * model_scope: name scope in which to define the model
        * enbl_dst: whether to create a model with distillation loss
        """

        # class-independent initialization
        super(FullPrecLearner, self).__init__(sm_writer, model_helper)

        # over-ride the model scope and distillation loss switch
        if model_scope is not None:
            self.model_scope = model_scope
        self.enbl_dst = enbl_dst if enbl_dst is not None else FLAGS.enbl_dst

        # class-dependent initialization
        if self.enbl_dst:
            self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)
        self.__build(is_train=True)
        self.__build(is_train=False)

    def train(self):
        """Train a model and periodically produce checkpoint files."""

        # initialization
        self.sess_train.run(self.init_op)
        if FLAGS.enbl_multi_gpu:
            self.sess_train.run(self.bcast_op)

        # train the model through iterations and periodically save & evaluate the model
        time_prev = timer()
        for idx_iter in range(self.nb_iters_train):
            # train the model
            if (idx_iter + 1) % FLAGS.summ_step != 0:
                self.sess_train.run(self.train_op)
            else:
                __, summary, log_rslt = self.sess_train.run([self.train_op, self.summary_op, self.log_op])
                if self.is_primary_worker('global'):
                    time_step = timer() - time_prev
                    self.__monitor_progress(summary, log_rslt, idx_iter, time_step)
                    time_prev = timer()

            # save & evaluate the model at certain steps
            if self.is_primary_worker('global') and (idx_iter + 1) % FLAGS.save_step == 0:
                self.__save_model(is_train=True)
                self.evaluate()

        # save the final model
        if self.is_primary_worker('global'):
            self.__save_model(is_train=True)
            self.__restore_model(is_train=False)
            self.__save_model(is_train=False)
            self.evaluate()

    def evaluate(self):
        """Restore a model from the latest checkpoint files and then evaluate it."""

        self.__restore_model(is_train=False)

        if FLAGS.factory_mode:
            tmp_image = scipy.misc.imread(FLAGS.data_dir_local + "/images/" + FLAGS.image_name)
            x, y, z = tmp_image.shape
            print(tmp_image.shape)
            size_low = FLAGS.input_size
            size_high = FLAGS.sr_scale * size_low

            coordx = x // size_low
            coordy = y // size_low
            nb_iters = int(np.ceil(float(coordy * coordx) / FLAGS.batch_size_eval))
            outputs = []
            # outputs_bic = []
            image = np.zeros([size_high * coordx, size_high * coordy, 3], dtype=np.uint8)
            # image_bic = np.zeros([size_high * coordx, size_high * coordy, 3], dtype=np.uint8)
            print(image.shape)
            print(nb_iters)
            for i in range(nb_iters):

                output = self.sess_eval.run(self.factory_op)
                for img in output[0]:
                    outputs.append(img)


            print(np.array(outputs).shape)
            index = 0
            for i in range(coordx):
                for j in range(coordy):
                    image[i*size_high: (i+1)*size_high, j*size_high: (j+1)*size_high, :] = np.array(outputs[index])
                    index += 1

            out = Image.fromarray(image, 'RGB')
            out.save('out_example/' + 'output.jpg')

            return

        nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size_eval))
        eval_rslts = np.zeros((nb_iters, len(self.eval_op)))

        # print("nb_iters: ", nb_iters)

        for idx_iter in range(nb_iters):
            eval_rslts[idx_iter] = self.sess_eval.run(self.eval_op)

        # eval_psnr = sorted(eval_rslts[:, 1])
        # for idx in range(nb_iters):
        #   print(eval_psnr[idx])

        for idx, name in enumerate(self.eval_op_names):
            tf.logging.info('%s = %.4e' % (name, np.mean(eval_rslts[:, idx])))

        t = time.time()
        for idx_iter in range(nb_iters):
            _ = self.sess_eval.run(self.time_op)

        t = time.time() - t
        images, outputs, labels = self.sess_eval.run(self.out_op)
        # print(labels[0])
        output_size = FLAGS.sr_scale * FLAGS.input_size
        for i in range(min(8, FLAGS.batch_size_eval)):
            img_bic = scipy.misc.imresize(images[i], (output_size, output_size), 'bicubic')
            img_bic = np.clip(img_bic, 0, 255)
            img_bic = np.array(img_bic, np.uint8)

            img_bic = Image.fromarray(img_bic, 'RGB')
            img = Image.fromarray(images[i], 'RGB')
            out = Image.fromarray(outputs[i], 'RGB')
            label = Image.fromarray(labels[i], 'RGB')
            img_bic.save(('out_example/' + str(i) + 'bic.jpg'))
            img.save('out_example/' + str(i) + 'image.jpg')
            out.save('out_example/' + str(i) + 'output.jpg')
            label.save('out_example/' + str(i) + 'label.jpg')

        tf.logging.info('time = %.4e' % (t / FLAGS.nb_smpls_eval))

        txt = open("log.txt", "a")
        l = ["full"]

        l += [self.model_name]
        # for idx, name in enumerate(self.eval_op_names):
        # tmp = np.mean(eval_rslts[:, 1])
        # l += ["PSNR: " + str(tmp)]
        for idx, name in enumerate(self.eval_op_names):
            tmp = np.mean(eval_rslts[:, idx])
            l += [name + ": " + str(tmp)]
        l += ["eval_batch_size: " + str(FLAGS.batch_size_eval)]
        l += ["time/pic: " + str(t / FLAGS.nb_smpls_eval)]

        txt.write(str(l))
        txt.write('\n')
        txt.close()


    def __build(self, is_train):  # pylint: disable=too-many-locals
        """Build the training / evaluation graph.

        Args:
        * is_train: whether to create the training graph
        """

        with tf.Graph().as_default():
            # TensorFlow session
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
            sess = tf.Session(config=config)

            # data input pipeline
            with tf.variable_scope(self.data_scope):
                iterator = self.build_dataset_train() if is_train else self.build_dataset_eval()
                images, labels = iterator.get_next()
                tf.add_to_collection('images_final', images)

            # model definition - distilled model
            if self.enbl_dst:
                logits_dst = self.helper_dst.calc_logits(sess, images)

            # model definition - primary model
            with tf.variable_scope(self.model_scope):
                # forward pass
                logits = self.forward_train(images) if is_train else self.forward_eval(images)
                # out = self.forward_eval(images)
                tf.add_to_collection('logits_final', logits)

                # loss & extra evalution metrics
                loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
                if self.enbl_dst:
                    loss += self.helper_dst.calc_loss(logits, logits_dst)
                tf.summary.scalar('loss', loss)
                for key, value in metrics.items():
                    tf.summary.scalar(key, value)

                # optimizer & gradients
                if is_train:
                    self.global_step = tf.train.get_or_create_global_step()
                    lrn_rate, self.nb_iters_train = setup_lrn_rate(
                        self.global_step, self.model_name, self.dataset_name)
                    optimizer = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
                    # optimizer = tf.train.AdamOptimizer(lrn_rate)
                    if FLAGS.enbl_multi_gpu:
                        optimizer = mgw.DistributedOptimizer(optimizer)
                    grads = optimizer.compute_gradients(loss, self.trainable_vars)

            # TF operations & model saver
            if is_train:
                self.sess_train = sess
                with tf.control_dependencies(self.update_ops):
                    self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
                self.summary_op = tf.summary.merge_all()
                self.log_op = [lrn_rate, loss] + list(metrics.values())
                self.log_op_names = ['lr', 'loss'] + list(metrics.keys())
                self.init_op = tf.variables_initializer(self.vars)
                if FLAGS.enbl_multi_gpu:
                    self.bcast_op = mgw.broadcast_global_variables(0)
                self.saver_train = tf.train.Saver(self.vars)
            else:
                self.sess_eval = sess

                self.factory_op = [tf.cast(logits, tf.uint8)]
                self.time_op = [logits]
                self.out_op = [tf.cast(images, tf.uint8), tf.cast(logits, tf.uint8), tf.cast(labels, tf.uint8)]
                self.eval_op = [loss] + list(metrics.values())
                self.eval_op_names = ['loss'] + list(metrics.keys())
                self.saver_eval = tf.train.Saver(self.vars)

    def __save_model(self, is_train):
        """Save the model to checkpoint files for training or evaluation.

        Args:
        * is_train: whether to save a model for training
        """

        if is_train:
            save_path = self.saver_train.save(self.sess_train, FLAGS.save_path, self.global_step)
        else:
            save_path = self.saver_eval.save(self.sess_eval, FLAGS.save_path_eval)
        tf.logging.info('model saved to ' + save_path)

    def __restore_model(self, is_train):
        """Restore a model from the latest checkpoint files.

        Args:
        * is_train: whether to restore a model for training
        """

        save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
        if is_train:
            self.saver_train.restore(self.sess_train, save_path)
        else:
            self.saver_eval.restore(self.sess_eval, save_path)
        tf.logging.info('model restored from ' + save_path)

    def __monitor_progress(self, summary, log_rslt, idx_iter, time_step):
        """Monitor the training progress.

        Args:
        * summary: summary protocol buffer
        * log_rslt: logging operations' results
        * idx_iter: index of the training iteration
        * time_step: time step between two summary operations
        """

        # write summaries for TensorBoard visualization
        self.sm_writer.add_summary(summary, idx_iter)

        # compute the training speed
        speed = FLAGS.batch_size * FLAGS.summ_step / time_step
        if FLAGS.enbl_multi_gpu:
            speed *= mgw.size()

        # display monitored statistics
        log_str = ' | '.join(['%s = %.4e' % (name, value)
                              for name, value in zip(self.log_op_names, log_rslt)])
        tf.logging.info('iter #%d: %s | speed = %.2f pics / sec' % (idx_iter + 1, log_str, speed))
