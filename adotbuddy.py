# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:18:29 2020

@author: Rahul Kumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import csv

df = pd.read_csv('train.csv')
fd = pd.read_csv('test.csv')
z = fd['pet_id']
#print(df.head())
#print(df['color_type'].values)

x = np.asarray(df[['length(m)', 'height(cm)', 'X1', 'X2']])
x1 =np.asarray(fd[['length(m)', 'height(cm)', 'X1', 'X2']])
#print(x[0:5])
y1 = np.asarray(df['breed_category'])
#print(y1[0:5])
y2 = np.asarray(df['pet_category'])
#print(y[0:5])
x = preprocessing.StandardScaler().fit(x).transform(x)
x1 = preprocessing.StandardScaler().fit(x1).transform(x1)
#print(x[0:5])

#from sklearn.model_selection import train_test_split

#xtrain, xtest, ytrain, ytest = train_test_split(x, y2, test_size =0.3, random_state =4)
#print (xtrain.shape, ytrain.shape)
#print(xtest.shape, ytest.shape)
from sklearn.neighbors import KNeighborsClassifier
k =3
neigh = KNeighborsClassifier(n_neighbors = k ).fit(x, y1)
predy = neigh.predict(x1)
#pred = np.array(predy)
k1 =5
neigh = KNeighborsClassifier(n_neighbors=k1).fit(x,y2)
pred = neigh.predict(x1)
row = ['pet_id', 'breed_category','pet_category']
with open('x.csv', 'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(row)
    for av in range(len(predy)):
        data = [z[av],predy[av],pred[av]]
        csvwriter.writerow(data)
#csvfile.close()
    

