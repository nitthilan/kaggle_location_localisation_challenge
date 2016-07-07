from itertools import islice
import matplotlib
import time
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import constants as ct
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GMM

import numpy as np
import classifier_functions as cf

matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


# Mostly not used file. 

def get_data(filename, is_train_data, scale_factor):
    NUM_FEATURES = 10
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        
        record_array = np.empty([num_entires, 4], dtype="f8")
        feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
        label = np.empty([num_entires, 1], dtype='f8').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [(float(x)) for x in data_element]
            record_array[row_id] = data_element[1:5]
            if(is_train_data):
                label[row_id] = data_element[5]
            row_id = row_id + 1
        
        feature[:,0] = scale_factor[0]*(record_array[:,0]/10000.0)
        feature[:,1] = scale_factor[1]*(record_array[:,1]/10000.0)
        feature[:,2] = scale_factor[2]*np.log10(record_array[:,2])*1.0

        time = record_array[:,3]
        minute = 2*np.pi*((time)%1440)/1440
        feature[:,3] = scale_factor[3]*(np.sin(minute)+1).round(4)
        feature[:,4] = scale_factor[3]*(np.cos(minute)+1).round(4)

        month = 2*np.pi*((time//1440)%30)/30
        feature[:,5] = scale_factor[4]*(np.sin(month)+1).round(4)#*0.01
        feature[:,6] = scale_factor[4]*(np.cos(month)+1).round(4)#*0.01
        
        day = 2*np.pi*((time//1440)%365)/365
        weekday = 2*np.pi*((time//1440)%7)/7
        feature[:,7] = scale_factor[5]*(np.sin(day)+1).round(4)
        feature[:,8] = scale_factor[5]*(np.cos(day)+1).round(4)

        #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
        feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001

    if(is_train_data):
        return (feature[:row_id], label[:row_id])
    else:
        return feature

def get_pred_actual_position(classifier, feature, label):
    predict = classifier.predict_proba(feature)
    pred_pos = np.argsort(predict, axis=1)
    classes = classifier.classes_
    inv_class_map = {}
    actual_idx = 0
    for idx in classes:
        inv_class_map[idx] = actual_idx;
        actual_idx += 1

    pred_cost_position = np.zeros((len(label), 4))
    for i in range(len(label)):
        act_label = label[i]
        pred_cost_position[i][0] = act_label
        if(inv_class_map.has_key(act_label)):
            label_mapped_to_class = inv_class_map[act_label]
            pred_cost_position[i][1] = label_mapped_to_class
            pred_pos_idx = len(classes) - np.where(pred_pos[i] == label_mapped_to_class)[0][0] - 1
            #print pred_pos_idx, pred_pos[i], label_mapped_to_class
            pred_cost_position[i][2] = pred_pos_idx
            pred_cost_position[i][3] = predict[i][label_mapped_to_class]
        else:
            pred_cost_position[i][1] = -1
            pred_cost_position[i][2] = -1
            pred_cost_position[i][3] = -1
    return pred_cost_position

def calculate_distance(distances):
    distances += 0.0000001
    return distances ** -2.75