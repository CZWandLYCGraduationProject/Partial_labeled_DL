import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1


def gcn(inputs, k, scope="", use_layer_pluse=True):
    '''
    Feature Map Increase to 16, it is because feature map to thin would cause calculation problem
    :return: [w, h, 16]
    '''
    scope = "gcn" + scope
    with tf.variable_scope(scope):
        left = slim.conv2d(inputs, 16, [1, k], activation_fn=None)
        left = slim.conv2d(left, 16, [k, 1], activation_fn=None)
        right = slim.conv2d(inputs, 16, [k, 1], activation_fn=None)
        right = slim.conv2d(right, 16, [1, k], activation_fn=None)
    if use_layer_pluse:
        return layer_plus(right, left)
    else:
        return right + left

def br(inputs, scope="", use_layer_pluse=True):
    scope = "br" + scope
    with tf.variable_scope(scope):
        right = slim.conv2d(inputs, 8, 3)
        right = slim.conv2d(right, 2, 3, activation_fn=None)
        left = slim.conv2d(inputs, 2, 1, activation_fn=None)
        if use_layer_pluse:
            rtn = layer_plus(right, left)
        else:
            rtn = right + left
    return rtn

def deconv(inputs, rate, k, scope=""):
    scope = "deconv" + scope
    with tf.variable_scope(scope):
        rtn = slim.conv2d_transpose(inputs, 2, k, stride=rate)
    return rtn

def layer_plus(layer_a, layer_b):
    ori_c = layer_b.shape[3]
    layer_c = tf.concat([layer_a, layer_b], 3)
    layer_c = slim.conv2d(layer_c, ori_c, 1, activation_fn=None)
    return layer_c