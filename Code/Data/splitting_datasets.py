import csv
import nltk
import os
import random
import collections
import pickle

'''












***** DO NOT RUN THIS OTHERWISE FUTURE RESULTS WILL BE INCONSISTENT!!!*****














This file splits the datasets up into training, validation, and test sets and
counts the number of times each word appears, recording them in dictionaries.
Then, these dictionaries are dumped to pickle files. The total number of 
lines in each of the sets is saved in a dictionary named counts.
Additionally, the headlines themselves are saved in pickle files.
'''
fake_file = open('clean_fake.txt')
real_file = open('clean_real.txt')

real_lines=[]
fake_lines=[]

for line in real_file:
    line=line.rstrip('\n')
    real_lines.append(line)
for line in fake_file:
    line=line.rstrip('\n')
    fake_lines.append(line)

    
#Randomly shuffle all the lines
random.shuffle(real_lines)
random.shuffle(fake_lines)

#Put the lines into three sets: test 70%, val 15%, test 15%
real_train_lines=real_lines[0:int(len(real_lines)*0.7)]
with open('real_train_lines.pickle', 'wb') as handle:
    pickle.dump(real_train_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
real_val_lines=real_lines[int(len(real_lines)*0.7):int(len(real_lines)*0.85)]
with open('real_val_lines.pickle', 'wb') as handle:
    pickle.dump(real_val_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
real_test_lines=real_lines[int(len(real_lines)*0.85):]
with open('real_test_lines.pickle', 'wb') as handle:
    pickle.dump(real_test_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
fake_train_lines=fake_lines[0:int(len(fake_lines)*0.7)]
with open('fake_train_lines.pickle', 'wb') as handle:
    pickle.dump(fake_train_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
fake_val_lines=fake_lines[int(len(fake_lines)*0.7):int(len(fake_lines)*0.85)]
with open('fake_val_lines.pickle', 'wb') as handle:
    pickle.dump(fake_val_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)
fake_test_lines=fake_lines[int(len(fake_lines)*0.85):]
with open('fake_test_lines.pickle', 'wb') as handle:
    pickle.dump(fake_test_lines, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Save the headline counts in pickle files
counts={'real_train':len(real_train_lines), 'real_val':len(real_val_lines),\
        'real_test':len(real_test_lines), 'fake_train':len(fake_train_lines), \
        'fake_val':len(fake_val_lines), 'fake_test':len(fake_test_lines)}
with open('counts.pickle', 'wb') as handle:
    pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Create dictionaries with the number of times each word appears
#Training:
words=[]
for line in real_train_lines:
    temp=line.split(' ')
    #Remove any duplicated words in the headline
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_train=collections.Counter(words)

words=[]
for line in fake_train_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_train=collections.Counter(words)

#Account for words that appear in one set but not the other
for key in real_train:
    if key not in fake_train:
        fake_train[key]=0
        
for key in fake_train:
    if key not in real_train:
        real_train[key]=0
        
        
#Validation:
words=[]
for line in real_val_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_val=collections.Counter(words)

words=[]
for line in fake_val_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_val=collections.Counter(words)

for key in real_val:
    if key not in fake_val:
        fake_val[key]=0
        
for key in fake_val:
    if key not in real_val:
        real_val[key]=0
        
words=[]
for line in real_test_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
real_test=collections.Counter(words)

words=[]
for line in fake_test_lines:
    temp=line.split(' ')
    unique=[]
    [unique.append(item) for item in temp if item not in unique]
    words.extend(unique)
fake_test=collections.Counter(words)

for key in real_test:
    if key not in fake_test:
        fake_test[key]=0
        
for key in fake_test:
    if key not in real_test:
        real_test[key]=0

with open('real_train.pickle', 'wb') as handle:
    pickle.dump(real_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('real_val.pickle', 'wb') as handle:
    pickle.dump(real_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('real_test.pickle', 'wb') as handle:
    pickle.dump(real_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_train.pickle', 'wb') as handle:
    pickle.dump(fake_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_val.pickle', 'wb') as handle:
    pickle.dump(fake_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('fake_test.pickle', 'wb') as handle:
    pickle.dump(fake_test, handle, protocol=pickle.HIGHEST_PROTOCOL)