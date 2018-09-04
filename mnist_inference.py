#coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

INPUT_NODE = 784 
OUTPUT_NODE = 10 

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一次卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
#第二次卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
#全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    #第一层卷积层
    with tf.variable_scope('layer1-conv1'):
        #创建过滤器权重变量和偏置项
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        #卷积
        conv1 = tf.nn.conv2d(
                input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        #去线性化
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        
    #第二层池化层
    with tf.name_scope('layer2_pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #第三层卷积层
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
                pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    #第四层池化层
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    #将第四层输出转化成第五层输入格式,拉直成一维数组
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #第五层全连接层
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接的权重需要加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    #第六层全连接层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    #返回第六层输出
    return logit





