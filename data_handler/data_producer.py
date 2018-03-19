import threading
import queue
from utils import *
import os
import cv2
import numpy as np
import random
import math
import bisect
import tensorflow as tf

class DataProducer:
    def __init__(self, config):
        self.crop_method = config["crop_method"]
        self.batch_size = config["batch_size"]
        self.pos_ratio = config["pos_ratio"]
        self.data_path = config["data_path"]
        self.flip_prob = config["flip_prob"]
        self.pos_data_path = config["pos_data_path"]
        self.neg_data_path = config["neg_data_path"]
        self.val_list = config["val_list"]
        self.pos_name = config["pos_name"]
        self.neg_name = config["neg_name"]
        self.label_path = config["label_path"]
        self.padding_size = config["padding_size"]
        self.patch_width = 512 + self.padding_size * 2
        self.rank_ratio = config["rank_ratio"]
        self.crop_step = config["crop_step"]
        self.img_mean = config["b_mean"], config["g_mean"], config["r_mean"]
        self.pos_weight = config["pos_weight"]
        self.white_ratio = config["white_ratio"]
        self.test_list = config["test_list"]
        self.test_data_path = config["test_data"]
        self.crop_img_num = (1536 // self.crop_step + 1) ** 2
        self.bags_per_batch = self.crop_img_num // self.batch_size + (self.crop_img_num % self.batch_size)

        self.train_q = queue.Queue(maxsize=2)
        self.test_q = queue.Queue(maxsize=(2 *self.bags_per_batch))
        self.val_q = queue.Queue(maxsize=(2 * self.bags_per_batch))

    def start(self):
        self.train_t = threading.Thread(target=self.train_run, daemon=True)
        self.val_t = threading.Thread(target=self.val_run, daemon=True)
        self.test_t = threading.Thread(target=self.test_run, daemon=True)
        self.train_t.start()
        self.val_t.start()
        # self.test_t.start()

    def train_run(self):
        '''
        Train Image Producing Pipeline , (train_image_batch, label _image_batch) in the Queue
        :return:
        '''
        pos_name_file = open(self.pos_name, 'r')
        neg_name_file = open(self.neg_name, 'r')
        val_name_file = open(self.val_list, 'r')
        val_name_list = [x.strip('\n') for x in val_name_file.readlines()]
        pos_name_list = [x.strip('\n') for x in pos_name_file.readlines()]
        neg_name_list = [x.strip('\n') for x in neg_name_file.readlines()]
        pos_cur = 0
        neg_cur = 0
        pos_image_num = int(self.batch_size * self.pos_ratio)
        neg_image_num = int(self.batch_size * (1 - self.pos_ratio))
        while True:
            # check whether image is in the val list
            pos_cur = pos_cur % len(pos_name_list)
            while pos_name_list[pos_cur] in val_name_list:
                pos_cur = (pos_cur + 1 )% len(pos_name_list)
            neg_cur = neg_cur % len(neg_name_list)
            while neg_name_list[neg_cur] in val_name_list:
                neg_cur = (neg_cur + 1) % len(neg_name_list)

            pos_image_name = os.path.join(self.pos_data_path, pos_name_list[pos_cur])
            neg_image_name = os.path.join(self.neg_data_path, neg_name_list[neg_cur])
            label_image_name = pos_name_list[pos_cur][:-4] + "png"
            label_image_name = os.path.join(self.label_path, label_image_name)
            pos_cur += 1
            neg_cur += 1
            # load image to the memory
            pos_image = cv2.imread(pos_image_name)
            neg_image = cv2.imread(neg_image_name)
            label_image = cv2.imread(label_image_name, -1)
            label_image = label_image[:, :, np.newaxis]
            # crop images and put them to the queue
            if self.crop_method == "normal":
                pos_crop_images = self.normal_crop(pos_image, step=self.crop_step)
                label_crop_images = self.normal_crop(label_image, step=self.crop_step)
                neg_crop_images = self.normal_crop(neg_image, step=self.crop_step)
                if not self.rank_ratio:
                    pos_index_array = np.random.choice(np.arange(0, self.crop_img_num), replace=False, size=pos_image_num)
                    neg_index_array = np.random.choice(np.arange(0, self.crop_img_num), replace=False, size=neg_image_num)
                else:
                    label_crop_images, pos_crop_images = self.rank_batch(label_crop_images, pos_crop_images)
                    if len(label_crop_images) < pos_image_num:
                        continue
                    pos_index_array = np.random.choice(np.arange(0, len(label_crop_images)), replace=False,
                                                       size=pos_image_num)
                    neg_index_array = np.random.choice(np.arange(0, len(label_crop_images)), replace=False,
                                                       size=neg_image_num)

                pos_batch = [pos_crop_images[x] for x in pos_index_array]
                label_batch = [label_crop_images[x] for x in pos_index_array] + [np.zeros(shape=label_crop_images[0].shape) for _ in range(neg_image_num)]
                neg_batch = [neg_crop_images[x] for x in neg_index_array]

                self.one_hot(label_batch)
                self.flip_images(pos_batch, label_batch)
                self.flip_images(neg_batch)

                image_batch = pos_batch + neg_batch
                self.normalize(image_batch)
                image_batch = np.array(pos_batch + neg_batch)
                label_batch = np.array(label_batch)
                self.train_q.put((image_batch, label_batch))
            elif self.crop_method == "ycl":
                pos_crop_images, label_crop_images = self.ycl_crop(pos_image, label_image)
                neg_crop_images = self.normal_crop(neg_image)
                neg_num = neg_image_num if len(pos_crop_images) >= pos_image_num else self.batch_size - len(pos_crop_images)
                random.shuffle(neg_crop_images)
                neg_batch = neg_crop_images[:neg_num]

                if len(pos_crop_images) > pos_image_num:
                    pos_batch = pos_crop_images[:pos_image_num]
                    label_batch = label_crop_images[:pos_image_num]
                else:
                    pos_batch = pos_crop_images
                    label_batch = label_crop_images
                label_batch += [np.zeros(shape=label_crop_images[0].shape) for _ in range(neg_num)]

                self.one_hot(label_batch)
                print(label_batch[0].shape)
                self.flip_images(pos_batch, label_batch)
                self.flip_images(neg_batch)

                image_batch = pos_batch + neg_batch
                self.normalize(image_batch)
                image_batch = np.array(pos_batch + neg_batch)
                label_batch = np.array(label_batch)
                self.train_q.put((image_batch, label_batch))
            elif self.crop_method == "noise":
                image_batch = np.random.rand(self.batch_size, 512 + 2 * self.padding_size, 512 + 2 * self.padding_size, 3)
                label_batch = np.random.rand(self.batch_size, 512 + 2 * self.padding_size, 512 + 2 * self.padding_size, 2)
                self.train_q.put((image_batch, label_batch))

    def test_run(self):
        test_name_file = open(self.test_list, 'r')
        test_names = [x.strip('\n') for x in test_name_file.readlines()]
        self.test_data_size = len(test_names)
        for i in range(len(test_names)):
            test_image_name = os.path.join(self.test_data_path, test_names[i])
            test_image = cv2.imread(test_image_name)
            image_crop = self.normal_crop(test_image, step=self.crop_step)
            self.normalize(image_crop)
            for i in range(self.bags_per_batch):
                bag = np.array(image_crop[self.batch_size* i:(i+1) * self.batch_size])
                self.train_q.put((bag, test_names[i]))

    def val_run(self):
        val_name_file = open(self.val_list, 'r')
        pos_name_file = open(self.pos_name, 'r')
        val_names = [x.strip('\n') for x in val_name_file.readlines()]
        self.val_data_size = len(val_names)
        pos_names = [x.strip('\n') for x in pos_name_file.readlines()]
        cur = 0
        while True:
            cur = cur % len(val_names)
            image_name = val_names[cur]
            cur += 1

            if image_name in pos_names:
                label_image_name = image_name[:-4] + 'png'
                label_image_name = os.path.join(self.label_path, label_image_name)
                label_image = cv2.imread(label_image_name, -1)
                label_image = label_image[:, :, np.newaxis]
                image_name = os.path.join(self.pos_data_path, image_name)
                image = cv2.imread(image_name)
            else:
                label_image = np.zeros((2048, 2048, 1))
                image_name = os.path.join(self.neg_data_path, image_name)
                image = cv2.imread(image_name)

            image_crop = self.normal_crop(image, step=self.crop_step)
            label_crop = self.normal_crop(label_image, step=self.crop_step)
            self.one_hot(label_crop)
            self.normalize(image_crop)
            for i in range(self.bags_per_batch):
                bag_imgs = image_crop[self.batch_size * i:(i + 1) * self.batch_size]
                bag_label = label_crop[self.batch_size * i:(i + 1) * self.batch_size]
                image_batch = np.array(bag_imgs)
                label_batch = np.array(bag_label)
                self.val_q.put((image_batch, label_batch))

    def consume(self, status):
        data = None
        if status == ModeStatus.TRAIN:
            data = self.train_q.get()
        elif status == ModeStatus.TEST:
            data = self.test_q.get()
        elif status == ModeStatus.VAL:
            data = self.val_q.get()
        return data

    def ycl_crop(self, image, label):
        rtnrects = self.generate_region(label)
        images_list = []
        label_list = []
        for i in range(len(rtnrects)):
            an_image = image[rtnrects[i][1]:rtnrects[i][1] + rtnrects[i][3],
                       rtnrects[i][0]:rtnrects[i][0] + rtnrects[i][3], :]
            a_label = label[rtnrects[i][1]:rtnrects[i][1] + rtnrects[i][3],
                       rtnrects[i][0]:rtnrects[i][0] + rtnrects[i][3], :]
            images_list.append(an_image)
            label_list.append(a_label)

        return images_list, label_list

    def generate_region(self, image):
        '''
        Generate box_list
        :param image:
        :return: a box_list which item is (x, y, width, height)
        '''
        image, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bd_boxes = [cv2.boundingRect(contours[i]) for i in range(len(contours))]
        rtn_box_list = []

        for i in range(len(bd_boxes)):
            gen_box_num = math.ceil(max(bd_boxes[i][3], bd_boxes[i][2]) / self.patch_width)
            x_points = self.generate_points(bd_boxes[i][0], bd_boxes[i][2], gen_box_num)
            y_points = self.generate_points(bd_boxes[i][1], bd_boxes[i][3], gen_box_num)
            random.shuffle(x_points)
            random.shuffle(y_points)
            tmp_box_list = []
            for j in range(len(x_points)):
                tmp_box_list.append([x_points[j] - self.padding_size, y_points[j] - self.padding_size,
                                     512 + self.padding_size * 2, 512 + self.padding_size * 2])
            self.legalize(tmp_box_list)
            rtn_box_list.extend(tmp_box_list)

        rtn_box_list = list(set(rtn_box_list))
        random.shuffle(rtn_box_list)
        return rtn_box_list

    def generate_points(self, begin ,interval, nums):
        sub_len = math.floor(interval / nums)
        point_list = []
        for i in range(nums):
            inner_step = random.randrange(0, 4, 1)
            point = (inner_step * 0.25 + i) * sub_len
            point_list.append(int(point + begin))
        return point_list

    def legalize(self, box_list):
        for i in range(len(box_list)):
            if box_list[i][0] < 0:
                box_list[i][0] = 0
            elif box_list[i][0] > 2048 - 512 - self.padding_size * 2:
                box_list[i][0] = 2048 - 512 - self.padding_size * 2
            if box_list[i][1] < 0:
                box_list[i][1] = 0
            elif box_list[i][1] > 2048 - 512 - self.padding_size * 2:
                box_list[i][1] = 2048 - 512 - self.padding_size * 2
            box_list[i] = tuple(box_list[i])
        box_list = list(set(box_list))

    def normal_crop(self, image, step=512):
        batch_images = []
        padding_para = ((self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0))
        for y in range(0, 2048, step):
            for x in range(0, 2048, step):
                if x + 512 > 2048 or y + 512 > 2048:
                    continue
                tmp_image = image[y:y + 512, x:x + 512]
                tmp_image = np.pad(tmp_image, padding_para, 'reflect')
                batch_images.append(tmp_image)
        return batch_images

    def flip_images(self, images, label_images=None):
        for i in range(len(images)):
            prob = random.random()
            if prob < self.flip_prob:
                images[i] = np.flip(images[i], 1)
                if label_images:
                    label_images[i] = np.flip(label_images[i], 1)

    def one_hot(self, labels):
        for i in range(len(labels)):
            a = np.zeros(shape=(labels[i].shape[0], labels[i].shape[1], 2))
            a[:, :, 0] = np.squeeze(labels[i] == 0, axis=2)
            a[:, :, 1] = np.squeeze(labels[i] > 0, axis=2) * self.pos_weight
            labels[i] = a

    def normalize(self, images):
        # normalize to [0, 1]
        for i in range(len(images)):
            images[i] = images[i].astype(np.float32)
            for j in range(images[i].shape[2]):
                images[i][:, :, j] -= self.img_mean[j]
            images[i] /= 255

    def get_valtest_size(self):
        return self.val_data_size, 0

    def rank_batch(self, label_batch, img_batch):
        scores = []
        for i in range(len(label_batch)):
            tmp = np.sum(label_batch[i] > 0) / label_batch[i].size
            scores.append(tmp)
        zip1 = zip(scores, label_batch, img_batch)
        zip1 = sorted(zip1, key=lambda x:x[0], reverse=True)
        scores, label_batch, img_batch = zip(*zip1)
        i = -1
        for g in range(len(scores)):
            if scores[g] < self.white_ratio:
                i = g
                break
        if i != -1:
            label_batch, img_batch = label_batch[:i], img_batch[:i]
        return label_batch, img_batch

    def get_bags_per_batch(self):
        return self.bags_per_batch
