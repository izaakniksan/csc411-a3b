import csv
import nltk
import os
import random
import collections
import pickle
from numpy import *
from sklearn.pipeline import Pipeline
from math import *
from sklearn import svm
from sklearn.linear_model import SGDClassifier

def create_v(set_words,all_words):
    '''
    Create input and expected output vectors for logistic regression. 
    set_words is array of cleaned headlines (which are separated into words)
    count=m
    len(all_words)=n
    all_words is all the words that appear in the trainingset
    '''
    
    #creating input vector and output vectors
    #v is input: nxm
 
    count=len(set_words)
    #create input:
    v=zeros((len(all_words),count)) 
    i=0
    for headline in set_words:
        for j in range(0,len(all_words)):
            if all_words[j] in headline:
                v[j][i]=1
            else:
                v[j][i]=0
        i=i+1
    return v

def clean_headline(headline,_set):
    #takes in a headline and removed duplicated words and any words not found 
    #in the set. Outputs a list of these words.
    line=headline
    line=line.rstrip('\n')
    temp=line.split(' ') 
    
    #remove any duplicated words in the headline:
    h_words=[] # h_words contains all the words in the headline
    [h_words.append(item) for item in temp if item not in h_words]

    #remove any words in the headline which are not found in the training set:
    [h_words.remove(word) for word in h_words if word not in _set]
    return h_words

def main():
    print('*** Part 1 running ***')    
    
    with open(os.getcwd()+'\Data\\real_train.pickle', 'rb') as handle:
        real_train = pickle.load(handle)  
    with open(os.getcwd()+'\Data\\real_val.pickle', 'rb') as handle:
        real_val = pickle.load(handle)   
    with open(os.getcwd()+'\Data\\real_test.pickle', 'rb') as handle:
        real_test = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_train.pickle', 'rb') as handle:
        fake_train = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_val.pickle', 'rb') as handle:
        fake_val = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_test.pickle', 'rb') as handle:
        fake_test = pickle.load(handle)
    with open(os.getcwd()+'\Data\\counts.pickle', 'rb') as handle:
        counts = pickle.load(handle)
    with open(os.getcwd()+'\Data\\real_train_lines.pickle', 'rb') as handle:
        real_train_lines = pickle.load(handle)
    with open(os.getcwd()+'\Data\\real_val_lines.pickle', 'rb') as handle:
        real_val_lines = pickle.load(handle)
    with open(os.getcwd()+'\Data\\real_test_lines.pickle', 'rb') as handle:
        real_test_lines = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_train_lines.pickle', 'rb') as handle:
        fake_train_lines = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_val_lines.pickle', 'rb') as handle:
        fake_val_lines = pickle.load(handle)
    with open(os.getcwd()+'\Data\\fake_test_lines.pickle', 'rb') as handle:
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
    
    #Create output vector y_val
    y_val=zeros((len(validationset_words),2))
    for i in range(0,len(real_val_lines)):
        y_val[i][0]=1 #real
    for i in range(len(real_val_lines),len(validationset_words)):
        y_val[i][1]=1 #fake
        
    y_val=y_val.T
    
    
#--------------------BUILD SVM CLASSIFIER-------------------------
    print('Creating SVM Classifier') 

    clf = svm.SVC()
    clf.fit(X, y)  

    
    
    
    print('*** Part 1 finished ***')    
    

if __name__ == "__main__":
    main()