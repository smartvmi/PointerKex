import os
import json
import glob
import time
import pickle
import numpy as np
import warnings
from matplotlib import pylab as plt
from plotly import express as px
from tqdm.notebook import tqdm
from time import time as timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve

from functions import get_dataset_file_paths, load_and_clean_heap, generate_dataset, test, get_data_for_testing

block_size=20000
root = "../Training/basic/V_7_0_P1/32"
#root = "../test"
model_path = './model.pkl'
 
train = False

if train:
    heap_paths, json_paths = get_dataset_file_paths(root)
    train_heap_paths, validate_heap_paths, train_json_paths, validate_json_paths = train_test_split(heap_paths, json_paths, test_size=0.01, random_state=42)
    start_time = timer()
    dataset, labels = generate_dataset(heap_paths=train_heap_paths, json_paths=train_json_paths, train_subset=True, block_size=block_size)
    end_time = timer()
    print('Completed dataset creation in %f seconds' % (end_time - start_time))
    print(len(dataset),len(labels))
    for i in range(0,len(dataset)):
        if labels[i]:
            print(dataset[i], labels[i])

    start_time = timer()
    clf = RandomForestClassifier(n_estimators=3)
    clf.fit(X=dataset, y=labels)
    end_time = timer()
    print('Completed training in %f seconds' % (end_time - start_time))

  # Save the model
    with open(model_path, 'wb') as fp:
        pickle.dump(clf, fp)

else:
    print('Testing Mode')
    with open(model_path, 'rb') as fp:
        clf = pickle.load(fp)




both_keys, one_key, zero_keys, total_files, found_key_count, total_individual_keys, one_key_found, no_key_found = get_data_for_testing(clf=clf, root=root)
print('Both Keys Found: ', both_keys)
print('One Key Found: ', one_key)
print('Zero Keys Found: ', zero_keys)
print('Total Files: ', total_files)
print('Total keys found', found_key_count)
print('Total individual keys', total_individual_keys)

with open('one_key_found.txt', 'w') as fp:
    fp.writelines(one_key_found)

with open('no_key_found.txt', 'w') as fp:
    fp.writelines(no_key_found)

