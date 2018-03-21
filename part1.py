import csv
import nltk
import os
import random
import collections
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from math import *
from sklearn import svm
from sklearn.linear_model import SGDClassifier

def main():
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    
if __name__ == "__main__":
    main()