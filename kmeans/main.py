import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import os
import time
from collections import Counter
matdata=io.loadmat('sets_CWRUdata_order')
kmeans = KMeans(n_clusters=10)  #构造聚类器
kmeans.fit(matdata['x_test'])   #训练聚类器

label_count= pd.Series(kmeans.labels_).value_counts() #统计各个类别的数目
print(label_count)
centroids =pd.DataFrame(kmeans.cluster_centers_) #找出聚类中心
print(centroids)
err = 0
y_test = []
for y in matdata['y_test'].tolist():
    y_test.append(y.index(1))

x_test = pd.DataFrame(matdata['x_test'])   
y_test = pd.DataFrame({"real_label": y_test})
labels =pd.DataFrame({"julei_label": kmeans.labels_}) #找出聚类中心
#print(labels)
new_df=pd.concat([x_test,y_test,labels],axis=1) 
#print(new_df)
new_df.to_csv("new_df.csv",index=False)
for i in range(10):
    #print('label '+str(i)+' :')
    new_df_albel = new_df[new_df['julei_label'] == i]['real_label']
    print(new_df_albel.value_counts())
    err += max(new_df_albel.value_counts())
print('accuracy:'+str(err*100/2000.)+'%')
    #print(new_df_albel.groupby(by='real_label').count())

pca = PCA(n_components=2)  # 建立pca model
new_pca = pd.DataFrame(pca.fit_transform(matdata['x_test']))  
#到底是给df还是new_df降维我很纠结,我觉得是df，但是df和new_df其实结果一样
# 可视化

# 可视化
plt.figure(figsize=(36,32))  
cluster = []
color = ['black','red','sienna','orange','yellow','green','blue','purple','y','tan']
for i in range(10):
    cluster.append(new_pca[new_df['julei_label'] == i])
    plt.scatter(cluster[i][0], cluster[i][1], c=color[i])

'''
cluster0 = new_pca[new_df['julei_label'] == 0]
plt.plot(cluster0[0], cluster0[1], 'bs')
cluster1 = new_pca[new_df['julei_label'] == 1]
plt.plot(cluster1[0], cluster1[1], 'go')
cluster2 = new_pca[new_df['julei_label'] == 2]
plt.plot(cluster2[0], cluster2[1], 'b*')
cluster3 = new_pca[new_df['julei_label'] == 3]
plt.plot(cluster3[0], cluster3[1], 'cs')
cluster4 = new_pca[new_df['julei_label'] == 4]
plt.plot(cluster4[0], cluster4[1], 'mo')
cluster5 = new_pca[new_df['julei_label'] == 5]
plt.plot(cluster5[0], cluster5[1], 'y*')
cluster6 = new_pca[new_df['julei_label'] == 6]
plt.plot(cluster6[0], cluster6[1], 'ks')
cluster7 = new_pca[new_df['julei_label'] == 7]
plt.plot(cluster7[0], cluster7[1], 'ro')
cluster8 = new_pca[new_df['julei_label'] == 8]
plt.plot(cluster8[0], cluster8[1], 'r*')
cluster9 = new_pca[new_df['julei_label'] == 9]
plt.plot(cluster9[0], cluster9[1], 'gs')
'''

plt.legend(['cluster0','cluster1','cluster2','cluster3','cluster4','cluster5','cluster6','cluster7','cluster8','cluster9']) #设置图例
plt.savefig('kmeans.png') #保存图片文件命名为






