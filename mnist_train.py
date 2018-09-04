#coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference

BATCH_SIZE = 100 #一个训练batch中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.08
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99 #滑动平均衰减率
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"

# 训练模型的过程
def train(mnist):
    #调整输入数据格式，输入为一个四维数组
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
        name='x-input')

    y_= tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    #计算l2正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, 1, regularizer)

    #定义存储训练轮数的变量,模拟神经网络中迭代的轮数，可以用于动态控制衰减率
    global_step = tf.Variable(0, trainable=False)
    #定义滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #定义一个更新变量滑动平均的操作列表
    variables_averages_op = variable_averages.apply(tf.trainable_variables()) 
    
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
                    LEARNING_RATE_BASE, 
                    global_step, 
                    mnist.train.num_examples / BATCH_SIZE,
                    LEARNING_RATE_DECAY,
                    staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    #一次完成通过反向传播来更新神经网络中的参数和更新每一个参数的滑动平均值的两个操作
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化Tensorflow持久类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        print ckpt
        if ckpt and ckpt.model_checkpoint_path:
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
        #训练
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) 
            
            #调整样本输入的格式
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    #声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()














