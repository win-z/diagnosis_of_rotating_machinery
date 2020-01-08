#coding:utf-8
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as scm
from vlib.layers import *
import tensorflow as tf
import numpy as np
from vlib.load_data import *
import os
import scipy.io as io
import numpy
import vlib.plot as plot
#import vlib.my_extract as dataload
import vlib.save_images as save_img
import time
from tensorflow.examples.tutorials.mnist import input_data #as mnist_data
mnist = input_data.read_data_sets('data/', one_hot=True)

# temp = 0.89
class Train(object):
    def __init__(self, sess, args):
        #sess=tf.Session()
        self.sess = sess
        self.img_size = 32   # the size of image
        self.trainable = True
        self.batch_size = 10  # must be even number 偶数
        self.lr = 0.0002
        self.mm = 0.5      # momentum term for adam
        self.z_dim = 128   # the dimension of noise z
        self.EPOCH = 300    # the number of max epoch
        self.LAMBDA = 0.1  # parameter of WGAN-GP
        self.model = args.model  # 'DCGAN' or 'WGAN'
        self.dim = 1       # RGB is different with gray pic
        self.num_class = 11
        self.load_model = args.load_model
        self.build_model()  # initializer
        self.ltrain_size=100
        self.ultrain_size=100
        self.test_size=1000
        self.datashape()
        self.real_train=150

        
    def datashape(self):
        matdata=io.loadmat('sets_self')
        cw_matdata=io.loadmat('sets_CWRUdata')
        perm0 = numpy.arange(matdata['x_train'].shape[0])
        perm1 = numpy.arange(matdata['x_test'].shape[0])
        #numpy.random.shuffle(perm0)
        #numpy.random.shuffle(perm1)
        self.x_train = (matdata['x_train'][perm0])[0:self.ltrain_size]
        self.x_ul_train = (matdata['x_train'][perm0])[self.ltrain_size:self.ltrain_size+self.ultrain_size]
        self.y_train = (matdata['y_train'][perm0])[0:self.ltrain_size]
        self.y_ul_train = (matdata['y_train'][perm0])[self.ltrain_size:self.ltrain_size+self.ultrain_size]
        self.x_test = (matdata['x_test'][perm1])[0:self.test_size]
        self.y_test = (matdata['y_test'][perm1])[0:self.test_size]
        
        self.cw_x_train = (cw_matdata['x_train'][perm0])[0:self.ltrain_size]
        self.cw_x_ul_train = (cw_matdata['x_train'][perm0])[self.ltrain_size:self.ltrain_size+self.ultrain_size]
        self.cw_y_train = (cw_matdata['y_train'][perm0])[0:self.ltrain_size]
        self.cw_y_ul_train = (cw_matdata['y_train'][perm0])[self.ltrain_size:self.ltrain_size+self.ultrain_size]
        self.cw_x_test = (cw_matdata['x_test'][perm1])[0:self.test_size]
        self.cw_y_test = (cw_matdata['y_test'][perm1])[0:self.test_size]


    def build_model(self):
        # build  placeholders
        self.x=tf.placeholder(tf.float32,shape=[self.batch_size,self.img_size*self.img_size*self.dim],name='real_img')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='noise')
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class - 1], name='label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='flag')
        self.flag2 = tf.placeholder(tf.float32, shape=[], name='flag2')

        # define the network
        self.G_img = self.generator('gen', self.z, reuse=False)  #生成的欺骗图片
        d_logits_r, layer_out_r = self.discriminator('dis', self.x, reuse=False)   #数据集经过判定器
        d_logits_f, layer_out_f = self.discriminator('dis', self.G_img, reuse=True)  #生成器图片经过判定器
        #####################################################？？？？？？？？？？？？？？？
        d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')  # D regular loss
        #d_regular = tf.add_n(tf.get_collection('diss'),'d_regular')
        #tf.add_n实现一个列表的元素的相加
        #tf.get_collection返回当前计算图中手动添加的张量集合
        #d_regular 表示正则化，这里将判别器中所有的weights做了l2正则。 
        # caculate the unsupervised loss
        un_label_r = tf.concat([tf.ones_like(self.label), tf.zeros(shape=(self.batch_size, 1))], axis=1) 
        #[batch_size,num_class]  前十位是1，最后一位是0     最后一位表示是r还是f的图片
        # CONCAT lianjielianggeshuzu 
        un_label_f = tf.concat([tf.zeros_like(self.label), tf.ones(shape=(self.batch_size, 1))], axis=1)
        logits_r, logits_f = tf.nn.softmax(d_logits_r), tf.nn.softmax(d_logits_f)
        #Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值
        #d_loss_r = -tf.log(tf.reduce_sum(logits_r[:, :-1])/tf.reduce_sum(logits_r[:,:]))
        #d_loss_f = -tf.log(tf.reduce_sum(logits_f[:, -1])/tf.reduce_sum(logits_f[:,:]))
        #d_loss_r = -tf.reduce_mean(tf.log((tf.reduce_sum(d_logits_r, axis=-1) - d_logits_r[:, -1])/tf.reduce_sum(d_logits_r,axis=-1))
        #d_loss_f = -tf.reduce_mean(tf.log((d_logits_f[:, -1])/tf.reduce_sum(d_logits_f,axis=-1)) 
        #d_loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_r*0.9, logits=d_logits_r))
        #求平均值                                                          [batch_size,num_class] [batch_size,num_class]
        #d_loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_f*0.9, logits=d_logits_f))
        #计算交叉熵
        d_loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_r*0.9, logits=tf.concat([tf.ones_like(self.label), tf.reshape(d_logits_r[:, -1],shape=[self.batch_size, 1])], axis=1) ))
        d_loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_f*0.9, logits=tf.concat([tf.zeros_like(self.label), tf.reshape(d_logits_f[:, -1],shape=[self.batch_size, 1])], axis=1) ))
        #d_loss_r = -tf.reduce_sum(logits_r*tf.log(logits_r))
        #d_loss_f = -tf.reduce_sum(logits_f*tf.log(logits_f))
        #计算交叉熵
        # feature match
        f_match = tf.constant(0., dtype=tf.float32)
        for i in range(4):
            f_match += tf.reduce_mean(tf.multiply(layer_out_f[i]-layer_out_r[i], layer_out_f[i]-layer_out_r[i]))
        #################################！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # caculate the supervised loss
        s_label = tf.concat([self.label, tf.zeros(shape=(self.batch_size,1))], axis=1)
        s_l_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=s_label*0.9, logits=d_logits_r))
        #s_l_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=un_label_f*0.9, logits=d_logits_f))  # same as d_loss_f
        self.d_l_1, self.d_l_2 = d_loss_r - d_loss_f, s_l_r
        #self.d_loss = d_loss_r + d_loss_f + s_l_r*self.flag*10 + d_regular
        self.d_loss = d_loss_r - d_loss_f + s_l_r*self.flag*10 + d_regular
        self.g_loss = d_loss_f + 0.01*f_match

        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]
        for v in all_vars:
            print (v)
        if self.model == 'DCGAN':
            self.opt_d = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'WGAN_GP':
            self.opt_d = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(1e-5, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        else:
            print ('model can only be "DCGAN","WGAN_GP" !')
            return
        # test
        test_logits, _ = self.discriminator('dis', self.x, reuse=True)
        test_logits = tf.nn.softmax(test_logits)
        temp = tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])
        #discriminator全连接输出的最后一位   很重要！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        for i in range(10):
                   #拼接向量
            temp = tf.concat([temp, tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])], axis=1)
        test_logits -= temp                           #axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。
        self.prediction = tf.nn.in_top_k(test_logits, tf.argmax(test_logits, axis=1), 1)
                          #tf.nn.in_top_k用于计算预测的结果和实际结果的是否相等，返回true或false
                          #最大值的位置标签对应onehot的1，相同则返回为true
        self.pre =  test_logits                
        self.saver = tf.train.Saver()
        if not self.load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif self.load_model:
            self.saver.restore(self.sess, os.getcwd()+'/model_saved/model.ckpt')
            print ('model load done')
        self.sess.graph.finalize()
        #结束当前的计算图，使之成为只读。

    def train(self):
        if not os.path.exists('model_saved'):
            os.mkdir('model_saved')
        if not os.path.exists('gen_picture'):
            os.mkdir('gen_picture')
        noise = np.random.normal(-1, 1, [self.batch_size, 128])
        temp = 0.45
        max_acc=[0,0,0,0,0]
        print ('training')
        for epoch in range(self.EPOCH):
            # iters = int(156191//self.batch_size)
            perm2 = numpy.arange(self.ltrain_size)
            numpy.random.shuffle(perm2)
            self.cw_x_train = (self.cw_x_train[perm2])[0:self.ltrain_size]
            self.cw_y_train = (self.cw_y_train[perm2])[0:self.ltrain_size]
            
            iters = (self.real_train)//self.batch_size
            flag2 = 1  # if epoch>10 else 0
            for idx in range(iters):
                start_t = time.time()                  #########################################？？？？
                flag = 1 if idx < (self.ltrain_size//self.batch_size) else 0 # set we use 2*batch_size=200 train data labeled.
                batchx, batchl = mnist.train.next_batch(self.batch_size,self.cw_x_ul_train,self.cw_y_ul_train,test=2)

                #print(batchx[:,0])
                # batchx, batchl = self.sess.run([batchx, batchl])
                g_opt = [self.opt_g, self.g_loss]
                d_opt = [self.opt_d, self.d_loss, self.d_l_1, self.d_l_2]
                feed = {self.x:batchx, self.z:noise, self.label:batchl,self.flag:flag, self.flag2:flag2}
                # update the Discrimater k times
                _, loss_d, d1,d2 = self.sess.run(d_opt, feed_dict=feed)
                # update the Generator one time
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                print ("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f, d1:%4f, d2:%4f"%
                       (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d, loss_g,d1,d2)), 'flag:',flag
                plot.plot('d_loss', loss_d)
                plot.plot('g_loss', loss_g)
                if ((idx+1) % 100) == 0:  # flush plot picture per 1000 iters
                    plot.flush()
                plot.tick()
                if (idx+1)%500==0:
                    print ('images saving............')
                    img = self.sess.run(self.G_img, feed_dict=feed)
                    save_img.save_images(img, os.getcwd()+'/gen_picture/'+'sample{}_{}.jpg'\
                                         .format(epoch, (idx+1)//500))
                    print ('images save done')
            test_acc = self.test()
            max_acc[0]=max(max_acc[0],test_acc)
            max_acc.sort()
            print(max_acc)
            plot.plot('test acc', test_acc)
            plot.flush()
            plot.tick()
            print ('test acc:{}'.format(test_acc), 'temp:%3f'%(temp))
            if test_acc > temp:
                print ('model saving..............')
                #path = os.getcwd() + '/model_saved'
                path = os.getcwd() + '/model_saved'
                save_path = os.path.join(path, "model.ckpt")
                self.saver.save(self.sess, save_path=save_path)
                print ('model saved...............')
                temp = test_acc

# output = conv2d('Z_cona{}'.format(i), output, 3, 64, stride=1, padding='SAME')

    def generator(self,name, noise, reuse):
        #把噪声逆卷积成图片
        with tf.variable_scope(name,reuse=reuse):
            l = self.batch_size   #50
            output = fc('g_dc', noise, 2*2*64)
            output = tf.reshape(output, [-1, 2, 2, 64])
            output = tf.nn.relu(self.bn('g_bn1',output))
            output = deconv2d('g_dcon1',output,5,outshape=[l, 4, 4, 64*4])
            output = tf.nn.relu(self.bn('g_bn2',output))
            #解卷积，放大尺寸
            output = deconv2d('g_dcon2', output, 5, outshape=[l, 8, 8, 64 * 2])
            output = tf.nn.relu(self.bn('g_bn3', output))

            output = deconv2d('g_dcon3', output, 5, outshape=[l, 16, 16,64 * 1])
            output = tf.nn.relu(self.bn('g_bn4', output))

            output = deconv2d('g_dcon4', output, 5, outshape=[l, 32, 32, self.dim])
            output = tf.image.resize_images(output, (32, 32))   #调整图像大小
            # output = tf.nn.relu(self.bn('g_bn4', output))
            return tf.nn.tanh(output)   #激活

    def discriminator(self, name, inputs, reuse):
        l = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (l,self.img_size,self.img_size,self.dim))
        with tf.variable_scope(name,reuse=reuse):
            out = []
            output = conv2d('d_con1',inputs,5, 64, stride=2, padding='SAME') #14*14 16*16
            output1 = lrelu(self.bn('d_bn1',output))
            out.append(output1)
            #*(i+2p-k)/s+1
            # output1 = tf.contrib.keras.layers.GaussianNoise
            #                          卷积核 大小，个数
            output = conv2d('d_con2', output1, 3, 64*2, stride=2, padding='SAME')#7*7 *128  8*8
            output2 = lrelu(self.bn('d_bn2', output))   #激活函数
            out.append(output2)
            output = conv2d('d_con3', output2, 3, 64*4, stride=1, padding='VALID')#5*5 *256 6*6
            output3 = lrelu(self.bn('d_bn3', output))
            out.append(output3)
            output = conv2d('d_con4', output3, 3, 64*4, stride=2, padding='VALID')#2*2 *256
            output4 = lrelu(self.bn('d_bn4', output))
            out.append(output4)
            output = tf.reshape(output4, [l, 2*2*64*4])# 2*2*64*4
            output = fc('d_fc', output, self.num_class)  #全连接层
            # output = tf.nn.softmax(output)
            return output, out

    def bn(self, name, input):
        #Batch Normalization通过减少内部协变量加速神经网络的训练。
        #可以用作conv2d和fully_connected的标准化函数。
        val = tf.contrib.layers.batch_norm(input, decay=0.9, 
                                           updates_collections=None,
                                           epsilon=1e-5,
                                           scale=True,
                                           is_training=True,
                                           scope=name)
        return val

    # def get_loss(self, logits, layer_out):
    def test(self):
        count = 0.
        print ('testing................')
        for i in range(self.test_size//self.batch_size):   #batch_size:50
            testx, textl = self.cw_x_test[i*self.batch_size:(i+1)*self.batch_size],self.cw_y_test[i*self.batch_size:(i+1)*self.batch_size]  #读测试集的输入和标签
            prediction = self.sess.run(self.prediction , feed_dict={self.x:testx, self.label:textl})
            pre = self.sess.run(self.pre , feed_dict={self.x:testx, self.label:textl})
            #print(prediction)
            print(pre)
            print(textl)
            count += np.sum(prediction)
        return (count+0.)/self.test_size

