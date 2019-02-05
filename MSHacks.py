#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:30:03 2018

@author: error404
"""


#Importing Datasets
import os
import numpy
import pickle
import pefile
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sklearn.ensemble as ek
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
#from keras.models import Sequential
#from keras.layers import Convolution2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense

# Tor

tor = pd.read_csv("torNonTor.csv",low_memory=False)

tor = tor.drop(tor.columns[[6,7]], axis=1)

f_tor = tor.iloc[:,4:-1].values
t_tor = tor.iloc[:,-1].values

lbl = LabelEncoder()
t_tor = lbl.fit_transform(t_tor)

scl = StandardScaler()
scl.fit(f_tor)
f_tor = scl.transform(f_tor)

f_tor_train, f_tor_test, t_tor_train, t_tor_test = train_test_split(f_tor,t_tor,test_size=0.2)

dtc = DecisionTreeClassifier()
dtc.fit(f_tor,t_tor)

t_tor_pred = dtc.predict(f_tor_test)

acc = accuracy_score(t_tor_test, t_tor_pred)
print("Accuracy for Tor classification is ")
print((acc*100).round(2))

# Botnet

botnet = pd.read_csv("Botnet.csv")
botnet = botnet.drop(botnet.columns[[20, 21]], axis=1)
f_botnet = botnet.iloc[:,7:-1].values
t_botnet = botnet.iloc[:,-1].values
lbl = LabelEncoder()
t_botnet = lbl.fit_transform(t_botnet)
scl = StandardScaler()
scl.fit(f_botnet)
f_botnet = scl.transform(f_botnet)
f_botnet_train, f_botnet_test, t_botnet_train, t_botnet_test = train_test_split(f_botnet,t_botnet,test_size=0.2)
dtc = DecisionTreeClassifier()
dtc.fit(f_botnet_train,t_botnet_train)
t_botnet_pred = dtc.predict(f_botnet_test)
acc = accuracy_score(t_botnet_test, t_botnet_pred)
print("Accuracy for Botnet Classification is ")
print((acc*100).round(2))

#  PortScan

rawData = pd.read_csv("PortScan.csv")
rawData = rawData.drop(rawData.columns[[20, 21]], axis=1)
feature = rawData.iloc[:,7:82].values
target = rawData.iloc[:,-1].values
encoder = LabelEncoder()
target = encoder.fit_transform(target)
sc = StandardScaler()
sc.fit(feature)
feature = sc.transform(feature)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2, random_state=2)
dtc = DecisionTreeClassifier()
dtc = dtc.fit(featureTrain, targetTrain)
targetPred = dtc.predict(featureTest)
acc = accuracy_score(targetTest,targetPred)*100
print("Accuracy for PortScan Classification is  ")
print(acc.round(2))

#DDoS

ddos = pd.read_csv("DDoS.csv",low_memory=False)
ddos = ddos.drop(ddos.columns[[20,21,85]], axis=1)
f_ddos = ddos.iloc[:,7:-1].values
t_ddos = ddos.iloc[:,-1].values
lbl = LabelEncoder()
t_ddos = lbl.fit_transform(t_ddos)
scl = StandardScaler()
scl.fit(f_ddos)
f_ddos = scl.transform(f_ddos)
f_ddos_train, f_ddos_test, t_ddos_train, t_ddos_test = train_test_split(f_ddos, t_ddos, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
knn.fit(f_ddos_train,t_ddos_train)
t_ddos_pred = knn.predict(f_ddos_test)
acc = accuracy_score(t_ddos_test, t_ddos_pred)
print("Accuracy for DDoS classification is ")
print((acc*100).round(2))

# TransferType

rawData = pd.read_csv("transfer_type.csv")
rawData = rawData.drop(rawData.columns[[0,1,2,3,4,7]], axis=1)
feature = rawData.iloc[:,2:22].values
target = rawData.iloc[:,-1].values
le = LabelEncoder()
target = le.fit_transform(target)
sc = StandardScaler()
sc.fit(feature)
feature = sc.transform(feature)
featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2, random_state=3)
dtc = DecisionTreeClassifier()
dtc = dtc.fit(featureTrain, targetTrain)
targetPred = dtc.predict(featureTest)
acc = accuracy_score(targetTest,targetPred)*100
print("Accuracy for TransferType Classification is  ")
print(acc.round(2))

####################### Malware Clasification ##################################
# ------------- This code requires tons of images tat cant be uploaded to Github or In this Machine  --------------------
#
## Initialising the CNN
#classifier = Sequential()
#
## Step 1 - Convolution
#classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
#
## Step 2 - Pooling
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Adding a second convolutional layer
#classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))
#
## Step 3 - Flattening
#classifier.add(Flatten())
#
## Step 4 - Full connection
#classifier.add(Dense(output_dim = 128, activation = 'relu'))
#classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#
## Compiling the CNN
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## Part 2 - Fitting the CNN to the images
#
#from keras.preprocessing.image import ImageDataGenerator
#
#train_datagen = ImageDataGenerator(rescale = 1./255,
#                                   shear_range = 0.2,
#                                   zoom_range = 0.2,
#                                   horizontal_flip = True)
#
#test_datagen = ImageDataGenerator(rescale = 1./255)
#
#training_set = train_datagen.flow_from_directory('dataset/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
#
#test_set = test_datagen.flow_from_directory('dataset/test_set',
#                                            target_size = (64, 64),
#                                            batch_size = 32,
#                                            class_mode = 'binary')
#
#classifier.fit_generator(training_set,
#                         samples_per_epoch = 8000,
#                         nb_epoch = 25,
#                         validation_data = test_set,
#                         nb_val_samples = 2000)

# Malware Classification

import pandas

dataset = pandas.read_csv('/home/error404/Downloads/data.csv',sep='|', low_memory=False)

dataset.groupby(dataset['legitimate']).size()

X = dataset.drop(['Name','md5','legitimate'],axis=1).values
y = dataset['legitimate'].values

extratrees = ek.ExtraTreesClassifier().fit(X,y)
model = SelectFromModel(extratrees, prefit=True)
X_new = model.transform(X)
nbfeatures = X_new.shape[1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y ,test_size=0.2)

features = []
index = numpy.argsort(extratrees.feature_importances_)[::-1][:nbfeatures]

for f in range(nbfeatures):
    print("%d. feature %s (%f)" % (f + 1, dataset.columns[2+index[f]], extratrees.feature_importances_[index[f]]))
    features.append(dataset.columns[2+f])

model = { "DecisionTree":tree.DecisionTreeClassifier(max_depth=10),
         "RandomForest":ek.RandomForestClassifier(n_estimators=50),
         "Adaboost":ek.AdaBoostClassifier(n_estimators=50),
         "GradientBoosting":ek.GradientBoostingClassifier(n_estimators=50),
         "GNB":GaussianNB(),
         "LinearRegression":LinearRegression()
}

results = {}
for algo in model:
    clf = model[algo]
    clf.fit(X_train,y_train)
    score = clf.score(X_test,y_test)
    print ("%s : %s " %(algo, score))
    results[algo] = score

winner = max(results, key=results.get)

joblib.dump(model[winner],'classifier/classifier.pkl')

open('classifier/features.pkl', 'w').write(pickle.dumps(features))

clf = model[winner]
res = clf.predict(X_new)
mt = confusion_matrix(y, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))

clf = joblib.load('classifier/classifier.pkl')

features = pickle.loads(open(os.path.join('classifier/features.pkl'),'r').read())
