#coding:utf-8
from glob import glob 
#文件操作相关的模块
#该方法返回所有匹配的文件路径列表，该方法需要一个参数用来指定匹配的路径字符串
from PIL import Image
#python中有关图像的库 因为这个程序是图像的应用 故障诊断应该用不到
import matplotlib.pyplot as plt
#python图像可视化的库
import scipy.misc as scm
#有两个作用 1.以图像形式保存数组 2.读写mat文件
from vlib.layers import *
# *表示的是任意文件 
# layers应该是各个层的定义 包括了卷积层、反卷积、全连接层、批标准化层、激活层....的定义

import tensorflow as tf
import numpy as np
from vlib.load_data import *
#加载数据用 
import os
import train
import vlib.my_extract as Dataload
#也是自己定义的 图像提取？？
import argparse
#命令行直接读取参数
parser = argparse.ArgumentParser()
#第一步：创建 ArgumentParser() 对象

parser.add_argument('--model', type=str, default='DCGAN', help='DCGAN or WGAN-GP')
parser.add_argument('--trainable', type=bool, default=False,help='True for train and False for test')
parser.add_argument('--load_model', type=bool, default=True, help='True for load ckpt model and False for otherwise')
parser.add_argument('--label_num', type=int, default=2, help='the num of labled images we use， 2*100=200，batchsize:100')
#第二步：调用 add_argument() 方法添加参数
#name or flags - 选项字符串的名字或者列表
#type - 命令行参数应该被转换成的类型
#default - 不指定参数时的默认值。
#help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息

args = parser.parse_args()
#第三步使用 parse_args() 解析添加的参数

def main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    #设置要使用的gpu内存 不然默认会占用所有的内存
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    #当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长

    sess = tf.InteractiveSession(config=config)
    #我们可以先构建一个session然后再定义操作（operation）
    model = train.Train(sess, args)
    #args 参数 
    #第一个参数：选择DCGAN模式还是WGAN-GP模式，二者的不同主要在于损失函数不同和优化器的学习率不同，其他都一样
    #第二个参数是args.trainable,训练还是测试，训练时为True，测试是False
    #第三个参数 表示是否选择加载训练好的权重
    #第四个参数 有标签的样本的数目
    if args.trainable:
        model.train()
    else:
        print model.test()

if __name__ == '__main__':
    main()
