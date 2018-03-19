import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import reduce
import numpy as np
import os

class DawnModel:
    def __init__(self, config):
        self.lr = config["lr"]
        self.model_save_path = config["model_save_path"]
        self.model_load_path = config["model_load_path"]
        self.momentum = config["momentum"]
        self.weight_decay = config["weight_decay"]
        self.lr_decay = config["lr_decay"]
        self.initialize()

    def initialize(self):
        self.load()
        self.inputs = self.graph.get_tensor_by_name("Input:0")
        self.global_step = self.graph.get_tensor_by_name("global_step:0")
        self.path0_output = self.graph.get_tensor_by_name("up_conv0_1/BiasAdd:0")
        # self.path1_output = self.graph.get_tensor_by_name("Conv_3/BiasAdd:0")
        # self.path2_output = self.graph.get_tensor_by_name("Conv_2/BiasAdd:0")
        # self.path3_output = self.graph.get_tensor_by_name("Conv_1/BiasAdd:0")
        # self.path4_output = self.graph.get_tensor_by_name("Conv/BiasAdd:0")

        self.path0_label = self.graph.get_tensor_by_name("path0_label:0")
        # self.path1_label = tf.image.resize_images(self.path0_label, [288, 288])
        # self.path2_label = tf.image.resize_images(self.path0_label, [144, 144])
        # self.path3_label = tf.image.resize_images(self.path0_label, [72, 72])
        # self.path4_label = tf.image.resize_images(self.path0_label, [36, 36])

        self.path0_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path0_label, logits=self.path0_output,
                                                                  name="path0_loss", dim=3)
        # self.path0_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path0_label, logits=self.path0_output,
        #                                                           name="path0_loss", dim=3)
        # self.path1_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path1_label, logits=self.path1_output,
        #                                                           name="path1_loss", dim=3)
        # self.path2_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path2_label, logits=self.path2_output,
        #                                                           name="path2_loss", dim=3)
        # self.path3_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path3_label, logits=self.path3_output,
        #                                                           name="path3_loss", dim=3)
        # self.path4_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.path4_label, logits=self.path4_output,
        #                                                           name="path4_loss", dim=3)

        #MARK Batch_size
        self.path0_loss = tf.reduce_sum(self.path0_loss / 4 / (576 * 576))
        # self.path1_loss = tf.reduce_sum(self.path1_loss / 4 / (288 * 288))
        # self.path2_loss = tf.reduce_sum(self.path2_loss / 4 / (144 * 144))
        # self.path3_loss = tf.reduce_sum(self.path3_loss / 4 / (72 * 72))
        # self.path4_loss = tf.reduce_sum(self.path4_loss / 4 / (36 * 36))

        self.lr = tf.train.exponential_decay(self.lr, self.global_step, self.lr_decay, 0.5)
        # self.total_loss = tf.reduce_sum([self.path0_loss, self.path1_loss, self.path2_loss, self.path3_loss, self.path4_loss]) / 5
        self.total_loss = self.path0_loss
        self.show_loss_value = self.total_loss
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        # a = tf.trainable_variables()
        a = slim.get_model_variables()

        #MARK BATCH_SIZE
        # self.optimize_target = self.total_loss + tf.contrib.layers.apply_regularization(self.l2_regularizer, a) / 8
        self.optimize_target = self.total_loss
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            try:
                self.optimize = self.optimizer.minimize(self.optimize_target, global_step=self.global_step)
                # print(self.optimize)
            except:
                self.optimize = self.graph.get_tensor_by_name("down_conv0/down_conv0_1/weights/Momentum:0")

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.show_loss_value)
            self.loss_summary = tf.summary.merge_all()

        self.prob_map = tf.nn.softmax(self.path0_output, dim=3)

    def load(self, model_load_path=None):
        if not model_load_path:
            model_load_path = self.model_load_path
        self.ckpt = tf.train.get_checkpoint_state(model_load_path)
        meta_graph_name = self.ckpt.model_checkpoint_path + ".meta"
        self.saver = tf.train.import_meta_graph(meta_graph_name)
        self.graph = tf.get_default_graph()

    def restore(self, sess, model_load_path=None):
        if not model_load_path:
            model_load_path = self.model_load_path
        self.saver.restore(sess, self.ckpt.model_checkpoint_path)

    def save(self, sess, model_save_path=None):
        if not model_save_path:
            model_save_path = self.model_save_path
        self.saver.save(sess, model_save_path, global_step=self.global_step)

    def dice_loss(self, out, loss, batch_size = 4):
        prob_map = tf.nn.softmax(out, dim=3)


