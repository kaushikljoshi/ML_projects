# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:47:54 2019

@author: kaushik Joshi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

data_name = "mnist"
data = io.loadmat("%s_data.mat" % data_name)
print("\nloaded %s data!" % data_name)
fields = "test_data", "training_data", "training_labels"
for field in fields:
    print(field, data[field].shape)

X = data['training_data']
y = data['training_labels']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.90, random_state = 3)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
#y_pred_train = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_train, y_pred_train)

from sklearn.metrics import accuracy_score
error = accuracy_score(y_test,y_pred_test)

