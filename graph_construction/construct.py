import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import graph_construction.graph_utils as gut
import numpy as np

BACKBONE_PATH = './model/basic/resnet_v1_50.ckpt'
# Input shape [batch, height, width, channels]

# Construct Graph
inputs = tf.placeholder(tf.float32, shape=(16, 576, 576, 3), name="Input")
print(inputs)
label = tf.placeholder(tf.float32, shape=(16, 576, 576, 2), name="path0_label")
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_50(inputs, is_training=True, reuse=None, global_pool=False,
                                             num_classes=None)
graph = tf.get_default_graph()
with tf.variable_scope("refine_path4"):
    path5_gcn = gut.gcn(net, 15)
    path5_gcn = gut.br(path5_gcn, scope='_0')
    path5_gcn_out = gut.deconv(path5_gcn, 2, 2)
    print(path5_gcn_out)

with tf.variable_scope("refine_path3"):
    block3_output = graph.get_tensor_by_name("resnet_v1_50/block3/unit_5/bottleneck_v1/Relu:0")
    path3 = gut.gcn(block3_output, 15)
    path3 = gut.br(path3, scope='_0')
    path3 = gut.layer_plus(path5_gcn_out, path3)
    path3 = gut.br(path3, scope='_1')
    path3_out = gut.deconv(path3, 2, 2)
    print(path3_out)

with tf.variable_scope("refine_path2"):
    block2_output = graph.get_tensor_by_name("resnet_v1_50/block2/unit_3/bottleneck_v1/Relu:0")
    path2 = gut.gcn(block2_output, 15)
    path2 = gut.br(path2, scope='_0')
    path2 = gut.layer_plus(path2, path3_out)
    path2 = gut.br(path2, scope='_1')
    path2_out = gut.deconv(path2, 2, 2)
    print(path2_out)

with tf.variable_scope("refine_path1"):
    block1_output = graph.get_tensor_by_name("resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0")
    path1 = gut.gcn(block1_output, 15)
    path1 = gut.br(path1, scope='_0')
    path1 = gut.layer_plus(path1, path2_out)
    path1 = gut.br(path1, scope='_1')
    path1_out = gut.deconv(path1, 2, 2)
    print(path1_out)

with tf.variable_scope("refine_path0"):
    path0 = path1_out
    path0 = gut.br(path0, scope='_0')
    path0 = gut.deconv(path0, 2, 2)
    path0 = gut.br(path0, scope='_1')
    print(path0)

global_step = tf.Variable(0, False, name="global_step")
random_inputs = np.random.rand(16, 576, 576, 3)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, "./model/version_1")