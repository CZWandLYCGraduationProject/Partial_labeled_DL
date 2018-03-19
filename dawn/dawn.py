import tensorflow as tf
from dawn.dawn_model import DawnModel
from data_handler.data_handler import DataHandler
from utils import *
import numpy as np
import sys
import os
import cv2

class DawnArchitect:
    def __init__(self, config):
        self.data_handler = DataHandler(config["data_policy"])
        self.batch_size = config["data_policy"]["batch_size"]
        self.pad_size = config["data_policy"]["padding_size"]
        self.bags_per_batch = self.data_handler.get_bags_per_batch()
        self.predict_save_path = config["predict_save_path"]
        self.crop_step = config["data_policy"]["crop_step"]
        self.threshold = config["architect_policy"]["threshold"]
        self.model = DawnModel(config["model_policy"])
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("./board_file")
        self.miu_list = []
        self.initialize()

    def initialize(self):
        self.model.initialize()
        self.model.restore(self.sess)
        global_initialize = tf.global_variables_initializer()
        self.sess.run(global_initialize)
        self.val_size, self.test_size = self.data_handler.get_valtest_size()

    def val(self):
        acu = 0
        name = 0
        for i in range(self.val_size):
            image =[]
            label = []
            for j in range(self.bags_per_batch):
                X_batch, Y_batch = self.data_handler.next_batch(ModeStatus.VAL)
                result = self.sess.run(self.model.prob_map, feed_dict={self.model.inputs:X_batch})
                image.append(result)
                label.append(Y_batch)
            image = np.concatenate(image, axis=0)
            label = np.concatenate(label, axis=0)
            result = self.reconstruct(image)
            label = self.reconstruct(label)
            result = (result[:, :, 1] > self.threshold).astype(np.uint8)
            label = label[:, :, 1].astype(np.uint8)
            # cv2.imwrite("./log/result/" + str(name) + ".png", result)
            name += 1
            acu += np.sum(result == label) / label.size
        acu /= self.val_size
        print("Validation ACU is {0}, global_step={1}".format(acu, self.model.global_step.eval(session=self.sess)))

    def predict(self):
        for i in range(self.test_size):
            image = []
            name = None
            for j in range(self.bags_per_batch):
                X_batch, name = self.data_handler.next_batch(ModeStatus.TEST)
                result = self.sess.run(self.model.prob_map, feed_dict={self.model.inputs:X_batch})
                image.append(result)
            name = name[:-4] + "png"
            image = np.concatenate(image, axis=0)
            image = self.reconstruct(image)
            image = (image[:, :, 1] > self.threshold).astype(np.uint8) * 255
            save_img_name = os.path.join(self.predict_save_path, name)
            cv2.imwrite(save_img_name, image)

    def train_a_step(self):
        X_batch, Y_batch = self.data_handler.next_batch(ModeStatus.TRAIN)
        _, loss, _  = self.sess.run([self.model.optimize, self.model.show_loss_value, self.model.loss_summary],
                                feed_dict={self.model.inputs:X_batch, self.model.path0_label:Y_batch})
        return loss

    def load(self, load_path=None):
        self.model.load(load_path)

    def save(self, save_path=None):
        self.model.save(self.sess, save_path)

    def get_miu (self, image, label):
        iou = 0
        for i in range(2):
            img, lab = image[:, :, i], label[:, :, i]

            if i == 0:
                img = img > (1 - self.threshold)
            else:
                img = img > self.threshold

            up, down = 0, 0
            img, lab = img.flatten(), lab.flatten()
            for i in range(img.size):
                if img[i] == 1 or lab[i] == 1:
                    down += 1
                if img[i] == 1 and lab[i] == 1:
                    up += 1
            iou += (up / (down + sys.float_info.epsilon))
        return iou / 2

    def get_accuracy (self, image, label):
        acu = np.sum(image == label)
        return acu

    def reconstruct(self, image):
        rtn_img = np.zeros((2048, 2048, 2), dtype=np.float)
        rtn_img.fill(float('-inf'))
        steps_per_row = ((2048 - 512) // self.crop_step) + 1
        for i in range(image.shape[0]):
            patch = image[i, :, :, :]
            patch = patch[self.pad_size:self.pad_size + 512, self.pad_size:self.pad_size + 512]
            x_pos, y_pos = self.crop_step * (i % steps_per_row), self.crop_step * (i // steps_per_row)
            img_patch = rtn_img[y_pos:y_pos + 512, x_pos:x_pos + 512]
            greater_index = np.greater(patch, img_patch)
            img_patch[greater_index] = patch[greater_index]
            rtn_img[y_pos:y_pos + 512, x_pos:x_pos + 512] = img_patch

        return rtn_img

    def get_setp(self):
        return self.model.global_step.eval(session=self.sess)
