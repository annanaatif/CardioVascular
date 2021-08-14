#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:05:41 2021

@author: aatifannan
"""

#This program classifies a person having cardiovascular disease or not

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Volumes/My Division/DS/cardio.csv')
df.head()

df.shape

df.describe

df['cardio'].value_counts()

sns.countplot(df['cardio'])

df['years'] = (df['age']/365).round(0)
df['years'] = pd.to_numeric(df['years'], downcast = 'integer')

sns.countplot(x='years', hue='cardio', data = df, palette= 'colorblind', edgecolor=sns.color_palette('dark', n_colors=1))

df.corr()

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
sns.heatmap(df.corr(), annot=True, fmt='.0%')

#drop years column
df=df.drop('years', axis=1)

df=df.drop('id', axis=1)

#splitting data into feature and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)
forest.fit(X_train, y_train)

#test model accuracy
model = forest
model.score(X_train, y_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print(cm)

print('Model Test Accuracy = {}'. format((TP+TN)/(TP+TN+FP+FN)))
