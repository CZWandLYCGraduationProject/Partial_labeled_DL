import tensorflow as tf
import tensorflow.contrib.slim as slim
import graph_construction.graph_utils as gut
import numpy as np
import time

inputs = tf.placeholder(tf.float32, shape=(None, 576, 576, 3), name = 'Input')
label = tf.placeholder(tf.float32, shape = (None, 576, 576, 2), name="path0_label")

# stride 1
path0 = slim.repeat(inputs, 2, slim.conv2d, 32, 3, scope='down_conv0', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path1 = slim.max_pool2d(path0, 2, scope='pool0')

# stride 2
path1 = slim.repeat(path1, 2, slim.conv2d, 64, 3, scope='down_conv1', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path2 = slim.max_pool2d(path1, 2, scope='pool1')

# stride 4
path2 = slim.repeat(path2, 2, slim.conv2d, 128, 3, scope='down_conv2', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path3 = slim.max_pool2d(path2, 2, scope='pool2')

# stride 8
path3 = slim.repeat(path3, 2, slim.conv2d, 256, 3, scope='down_conv3', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path4 = slim.max_pool2d(path3, 2, scope='pool3')

# stride 16
path4 = slim.repeat(path4, 2, slim.conv2d, 512, 3, scope='up_conv4', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path4_out = slim.conv2d(path4, 2, 3, activation_fn=None)

up_path4 = slim.conv2d_transpose(path4, 256, 2, 2, activation_fn=None, normalizer_fn=slim.batch_norm)

# up_p3p4 = gut.layer_plus(up_path4, path3)
up_p3p4 = tf.concat([up_path4, path3], 3)
up_p3p4 = slim.repeat(up_p3p4, 2, slim.conv2d, 256, 3, scope='up_conv3', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path3_out = slim.conv2d(up_p3p4, 2, 3, activation_fn=None)
up_p3 = slim.conv2d_transpose(up_p3p4, 128, 2, 2, activation_fn=None, normalizer_fn=slim.batch_norm)

up_p2p3 = tf.concat([up_p3, path2], 3)
#up_p2p3 = gut.layer_plus(up_p3, path2)
up_p2p3 = slim.repeat(up_p2p3, 2, slim.conv2d, 128, 3, scope='up_conv2', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path2_out = slim.conv2d(up_p2p3,2,3, activation_fn=None)
up_p2 = slim.conv2d_transpose(up_p2p3, 64, 2, 2, activation_fn=None, normalizer_fn=slim.batch_norm)

up_p1p2 = tf.concat([up_p2, path1], 3)
up_p1p2 = slim.repeat(up_p1p2, 2, slim.conv2d, 64, 3, scope='up_conv1', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path1_out = slim.conv2d(up_p1p2, 2, 3, activation_fn=None)
up_p1 = slim.conv2d_transpose(up_p1p2, 32, 2, 2, activation_fn=None, normalizer_fn=slim.batch_norm)

up_p0 = tf.concat([up_p1, path0], 3)
#up_p0 = gut.layer_plus(up_p1, path0)
up_p0 = slim.repeat(up_p0, 2, slim.conv2d, 32, 3, scope='up_conv0', normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
path0_out = slim.conv2d(up_p0, 2, 3, scope='up_conv0', activation_fn=None)
global_step = tf.Variable(0, False, name="global_step")

saver = tf.train.Saver()

random_input = np.random.rand(8, 576, 576, 3)
random_label = np.random.rand(8, 576, 576, 2)

# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=path0_out, name="path0_loss")
# loss = tf.reduce_sum(loss) / 4
#
# optimize = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(loss)
print(path0_out)
print(path1_out)
print(path2_out)
print(path3_out)
print(path4_out)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # t1 = time.time()
    # for _ in range(10):
    #     losss, _ = sess.run([loss, optimize], feed_dict={label:random_label, inputs:random_input})
    #     print(losss)
    # t2 = time.time()
    # print("Time is ", t2 - t1)
    saver.save(sess, './graph_construction/unet/unet_small')

