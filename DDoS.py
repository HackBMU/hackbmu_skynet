#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 27 18:18:51 2018

@author: Kashyap Patel
"""
 

#importing Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


############################  Botnet Classification  ########################
rawData_Bot = pd.read_csv('Botnet.csv')
rawData_Bot = rawData_Bot.drop(rawData_Bot.columns[[20, 21]], axis=1)

feature = rawData_Bot.iloc[:,7:82].values
target = rawData_Bot.iloc[:,-1].values

encoder = LabelEncoder()
target = encoder.fit_transform(target)

sc = StandardScaler()
sc.fit(feature)
feature = sc.transform(feature)

featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2)

dtc = DecisionTreeClassifier()
dtc = dtc.fit(featureTrain, targetTrain)
targetPred = dtc.predict(featureTest)

acc = accuracy_score(targetTest,targetPred)*100
print("Accuracy of DTC's algorithm for Botnet Classification is  ")
print(acc.round(2))


#########################  DDoS Classification  #######################
rawData_DDoS = pd.read_csv("DDoS.csv",low_memory=False)
rawData_DDoS = rawData_DDoS.drop(rawData_DDoS.columns[[20, 21, 85]], axis=1)

feature = rawData_Bot.iloc[:,7:82].values
target = rawData_Bot.iloc[:,-1].values

encoder = LabelEncoder()
target = encoder.fit_transform(target)

sc = StandardScaler()
sc.fit(feature)
feature = sc.transform(feature)

featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2, random_state=2)

knn = KNeighborsClassifier(n_neighbors=5,algorithm='auto')
knn.fit(featureTrain,targetTrain)
targetPred = knn.predict(featureTest)

acc = accuracy_score(targetTest,targetPred)*100
print("Accuracy of KNN's algorithm for DDoS classfications is  ")
print(acc.round(2))


#####################  PortScanning Classification  ####################
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
print("Accuracy of DTC's algorithm for PortSacnnig Classification is  ")
print(acc.round(2)) 


######################  TransferType Classifier #######################
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
print("Accuracy of DTC's algorithm for TransferType Classification is  ")
print(acc.round(2)) 


#######################  Tor-NonTor Classification ########################
rawData = pd.read_csv("torNonTor.csv",low_memory=False)
rawData = rawData.drop(rawData.columns[[6,7]], axis=1)

feature = rawData.iloc[:,5:-1].values
target = rawData.iloc[:,-1].values

featureTrain, featureTest, targetTrain, targetTest = train_test_split(feature, target, test_size=0.2, random_state=3)

dtc = DecisionTreeClassifier()
dtc = dtc.fit(featureTrain, targetTrain)
targetPred = dtc.predict(featureTest)

acc = accuracy_score(targetTest,targetPred)*100
print("Accuracy of DTC's algorithm for Tor-NonTor Classification is  ")
print(acc.round(2)) 
