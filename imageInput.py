#coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from PIL import Image
from pylab import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
pic_path = 'numbers/'

import mnist_inference
import mnist_train

#图片预处理
def pre_pic(picName):
    img = Image.open(picName) #打开图片
    reIm = img.resize((28,28), Image.ANTIALIAS) #将图片大小缩小至28*28的大小
    reIm = reIm.convert('L') #转换成黑白灰度级的图片
    #reIm.show() #显示图片
    im_arr = np.array(reIm) #将图片转换成数组的形式
    im_arr = np.ones((28,28))*255 - im_arr #图片颜色取反
    im_arr = im_arr.reshape([1, 784]) #将图片转换成1*784的一维数组
#    im_arr = im_arr.flatten()
    im_arr = im_arr.astype(np.float32) #将数组中元素转换为float32类型，与测试中的输入类型对应
    #二值化处理
    im_arr[im_arr < 50] = 0
    im_arr[im_arr >= 50] = 1
    #二值化处理 error
#    for i in range(784):
#        if im_arr[i] >= 10.:
#            im_arr[i] = 1.
#        else:
#            im_arr[i] = 0.
    print im_arr
    print im_arr.shape
    return im_arr


def restore_model(testPicArr):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE])
        y = mnist_inference.inference(x, None)
        preValue = tf.argmax(y, 1)
   
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                print preValue
            else:
                print('No checkpoint file found')
                return 

def main(argv):
 #  path = input("Please input the picture: ")
    picName = pic_path + argv[1]
    print picName
    testPicArr = pre_pic(picName)
    restore_model(testPicArr)
    
if __name__ == '__main__':
    tf.app.run()

