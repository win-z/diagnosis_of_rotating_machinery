# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:20:35 2019

@author: Liqi|leah0o
"""
import numpy
import numpy as np 
import xlrd
import scipy.io as io
from sklearn.preprocessing import normalize as norm
from scipy.fftpack import fft
from os.path import splitext
 


def shuru(data):
    """
    load data from xlsx
    """
    datatype=splitext(data)[1]  #获取后缀名

    if datatype =='.xlsx':
        excel=xlrd.open_workbook(data)   #打开表格
        sheet=excel.sheets()[1]   #sheet1
        data=sheet.col_values(0)  #读取第一列全部内容

    if datatype == '.mat':
        matdata=io.loadmat(data)
        fliter_=filter(lambda x: 'DE_time' in x, matdata.keys())  
        #lambda是一个匿名函数 filter过滤序列，保留字典中key为DE_time的值
        fliter_list = [item for item in fliter_]  #复制fliter给flliter_list
        idx=fliter_list[0]  #第一个item？
        data=matdata[idx][:, 0]  #切片遍历进行复制
    return data

def meanstd(data):      #归一化输入   Z-score标准化
    """
    to -1~1不用了，还是用sklearn的比较方便
    """
    for i in range(len(data)):
        datamean=np.mean(data[i])  #平均值
        datastd=np.std(data[i])    #计算标准差
        data[i]=(data[i]-datamean)/datastd
    
    return data

def sampling(data_this_doc,num_each,sample_lenth):
    """
    input:
        文件地址
        训练集的数量
        采样的长度
        故障的数量
    output:
        采样完的数据
    shuru->取长度->归一化
    ------
    note：采用的normalization 真这个话，除以l2范数
    """
    
    temp=shuru(data_this_doc)  #shuru函数读取文件里的内容

   
    #随机采样  为什么*2？
    idx = np.random.randint(0, len(temp)-sample_lenth*2, num_each)
    temp_sample=[]
    for i in range(num_each):
        time=temp[idx[i]:idx[i]+sample_lenth*2]
        fre=abs(fft(time))[0:sample_lenth]     #fft傅里叶变换    采了两个长度但是只要了一个长度
        fre = fre[::1]
        temp_sample.append(fre) 
    #
    #   有num_each*temp_sample长度的数据   #输入转为向量模式
    temp_sample=norm(temp_sample)       ##########求矩阵的1范数  （列模，列绝对值之和的最大值，一个矩阵的1范数是一个一维的数）
    #向量的2范数（平方和开平方根l1）  每个元素除以l1
    
    #temp_sample = meanstd(temp_sample)
    return temp_sample  #[num_each,sample_lenth]

class readdata():
    '''
    连接数据集、连接标签、输出
    '''
    def __init__(self,data_doc,num_each=400,ft=10,sample_lenth=1024):
        self.data_doc=data_doc
        ###特殊的再计算
        self.num_train=num_each*ft###       #不太清楚这里为什么要初始化ft=2？        ft=len(train_data_name)
        
        self.ft=ft
        self.sample_lenth=sample_lenth
        self.row=num_each
    
    def concatx(self):
        """
        连接多个数据
        暂且需要有多少数据写多少数据
        """
        #最后给的data_doc是train_data_name or test_data_name
        #该函数的作用是对列出的数据集分别采样并进行拼接
        data=np.zeros((self.num_train,self.sample_lenth))
        for i,data_this_doc in enumerate(self.data_doc):
            data[0+i*self.row:(i+1)*self.row]=sampling(data_this_doc,self.row,self.sample_lenth)   #num_train*1大小
        return data  #[len(enumerate(self.data_doc)),num_each,sample_lenth]

    def labelling(self):   
        """
        根据样本数和故障类型生成样本标签
        one_hot
        """
    
        label=np.zeros((self.num_train,self.ft))
        for i in range(self.ft):

            label[0+i*self.row:self.row+i*self.row,i]=1
#1 0 0 0 0 0 ...   宽度ft |
#1 0 0 0 0 0..            |长度num_each
#.............
   #1...
      #1...
         #1...
#ft种 ont_hot
        return label   #[num_train,self.ft]
       
    def output(self):
        '''
        输出数据集的数据和标签
        '''
        data=self.concatx()
        
        label=self.labelling()
        size=int(float(self.sample_lenth)**0.5)    #开方，应该是为了变成正方形图像矩阵
        data=data.astype('float32').reshape(self.num_train,1,size,size)
        label=label.astype('float32')  #变量类型转换
        return data,label
    

def dataset(train_data_name,data_name='sets',num_each=400,sample_lenth=1024,test_rate=0.5):
    '''
    根据特定的数据集构建
    '''

    test_data_name=train_data_name
    
    

    if test_rate==0:
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)
        
        
        x_test,y_test=testingset.output()
        io.savemat(data_name,{'x_test': x_test,'y_test': y_test,})
        return x_test,y_test
    else:
        
        trainingset=readdata(train_data_name,
                             ft=len(train_data_name),
                             num_each=num_each,
                             sample_lenth=sample_lenth)
        
        testingset=readdata(test_data_name,
                             ft=len(train_data_name),
                             num_each=int(num_each*test_rate),
                             sample_lenth=sample_lenth)
        ft=len(train_data_name)
        x_train,y_train=trainingset.output()
        perm0 = numpy.arange(x_train.shape[0])
        numpy.random.shuffle(perm0)
        x_train=x_train.reshape((num_each*ft,1024))
        y_train=y_train.reshape((num_each*ft,10))
        x_train=x_train[perm0]
        y_train=y_train[perm0]
        
        x_test,y_test=testingset.output()
        perm1 = numpy.arange(x_test.shape[0])
        numpy.random.shuffle(perm1)
        x_test=x_test.reshape((int(num_each*test_rate*ft),1024))
        y_test=y_test.reshape((int(num_each*test_rate*ft),10))
        x_test=x_test[perm1]
        y_test=y_test[perm1]
        io.savemat(data_name,{'x_train': x_train,'y_train': y_train,'x_test': x_test,'y_test': y_test,})
        return x_train,y_train,x_test,y_test
        #数据没有随机打乱，按照one_hot的顺序
#十个数据是分别按照故障类型选的吗？
#测试集和训练集完全来自同一数据集
def datashape():
    matdata=io.loadmat('sets_CWRUdata')
    return matdata['x_train'].reshape((matdata['x_train'].shape[0],1024)),matdata['y_train'].reshape((matdata['y_train'].shape[0],10)),matdata['x_test'].reshape((matdata['x_test'].shape[0],1024)),matdata['y_test'].reshape((matdata['y_test'].shape[0],10))
if __name__ == "__main__":
    train_data_name=[       
                      '自家试验台数据/012/4.xlsx',
                      '自家试验台数据/030/4.xlsx',
                      '自家试验台数据/055/4.xlsx',
                      '自家试验台数据/008/4.xlsx',
                      '自家试验台数据/035/4.xlsx',
                      '自家试验台数据/059/4.xlsx',
                      '自家试验台数据/016/4.xlsx',
                      '自家试验台数据/039/4.xlsx',
                      '自家试验台数据/063/4.xlsx',
                      '自家试验台数据/004/4.xlsx'
                      ]
    train_data_name_target=[     
                      '自家试验台数据/011/4.xlsx',
                      '自家试验台数据/030/4.xlsx',
                      '自家试验台数据/054/4.xlsx',
                      '自家试验台数据/007/4.xlsx',
                      '自家试验台数据/034/4.xlsx',
                      '自家试验台数据/058/4.xlsx',
                      '自家试验台数据/015/4.xlsx',
                      '自家试验台数据/038/4.xlsx',
                      '自家试验台数据/062/4.xlsx',
                      '自家试验台数据/003/4.xlsx'
                      ]
    CWRUdata=[
              '西储/106.mat',
              '西储/170.mat',
              '西储/210.mat',
              '西储/145.mat',
              '西储/309.mat',
              '西储/236.mat',
              '西储/121.mat',
              '西储/223.mat',
              '西储/186.mat',
              '西储/98.mat'
              ]
    #dataset(train_data_name_target,data_name='sets_self1')
    #dataset(train_data_name,data_name='sets_self2')
    #dataset(CWRUdata,data_name='sets_CWRUdata')
    matdata=io.loadmat('sets_self1')
    print(matdata['x_test'].shape)
    print(matdata['y_test'])
    matdata=io.loadmat('sets_self2')
    print(matdata['y_test'].shape)
    