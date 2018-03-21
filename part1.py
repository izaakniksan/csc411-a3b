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
    print('*** Part 1 running ***')    
    
    with open('real_train.pickle', 'rb') as handle:
        real_train = pickle.load(handle)  
    with open('real_val.pickle', 'rb') as handle:
        real_val = pickle.load(handle)   
    with open('real_test.pickle', 'rb') as handle:
        real_test = pickle.load(handle)
    with open('fake_train.pickle', 'rb') as handle:
        fake_train = pickle.load(handle)
    with open('fake_val.pickle', 'rb') as handle:
        fake_val = pickle.load(handle)
    with open('fake_test.pickle', 'rb') as handle:
        fake_test = pickle.load(handle)
    with open('counts.pickle', 'rb') as handle:
        counts = pickle.load(handle)
    with open('real_train_lines.pickle', 'rb') as handle:
        real_train_lines = pickle.load(handle)
    with open('real_val_lines.pickle', 'rb') as handle:
        real_val_lines = pickle.load(handle)
    with open('real_test_lines.pickle', 'rb') as handle:
        real_test_lines = pickle.load(handle)
    with open('fake_train_lines.pickle', 'rb') as handle:
        fake_train_lines = pickle.load(handle)
    with open('fake_val_lines.pickle', 'rb') as handle:
        fake_val_lines = pickle.load(handle)
    with open('fake_test_lines.pickle', 'rb') as handle:
        fake_test_lines = pickle.load(handle)

    all_words=[]
    [all_words.append(word) for word in real_train.keys()]
    
    print('Creating input vectors') 
#--------------------CREATE SETS,INPUTS,OUTPUTS-------------------------
    print('creating network inputs')
    #First create trainingset
    trainingset=append(real_train_lines,fake_train_lines)

    #Make arrays of words for trainingset
    trainingset_words=[]
    for i in range(0,len(trainingset)):
        trainingset_words.append(clean_headline(trainingset[i],trainingset))

    #Create input vector v: nxm=len(all_words) x count

    v_train=create_v(trainingset_words,all_words)
    v_train = vstack((ones((1, v_train.shape[1])), v_train))
    
    #Create output vector y
    #y is output: jxm=2xm = #possible outputs (real or fake) x #examples 
    y_train=zeros((len(trainingset_words),2))
    for i in range(0,len(real_train_lines)):
        y_train[i][0]=1 #real
    for i in range(len(real_train_lines),len(trainingset_words)):
        y_train[i][1]=1 #fake

    y_train=y_train.T #used because I did loop indices wrong
    
    #Create validation set
    validationset=append(real_val_lines,fake_val_lines)
    
    #Make arrays of words for validation set
    validationset_words=[]
    for i in range(0,len(validationset)):
        validationset_words.append(clean_headline(validationset[i],trainingset))
        
    #Create v_val
    
    v_val=create_v(validationset_words,all_words)
    v_val = vstack((ones((1, v_val.shape[1])), v_val))
    
    #Create output vector y_val
    y_val=zeros((len(validationset_words),2))
    for i in range(0,len(real_val_lines)):
        y_val[i][0]=1 #real
    for i in range(len(real_val_lines),len(validationset_words)):
        y_val[i][1]=1 #fake
        
    y_val=y_val.T
    
    
#--------------------BUILD SVM CLASSIFIER-------------------------
    print('Creating SVM Classifier') 
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)  
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    
    
    
    print('*** Part 1 finished ***')    
    

if __name__ == "__main__":
    main()