#!/usr/bin/python
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
from sklearn import preprocessing

import numpy as np
import classifier_functions as cf

from bayes_opt import BayesianOptimization

matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

def get_pred_actual_position(predict, classes, feature, label):
    pred_pos = np.argsort(predict, axis=1)
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
    (knc_n, bins, patches) = plt.hist(pred_cost_position[:,2], bins=np.max(pred_cost_position[:,2])+1)
    return knc_n

GLOBAL_POWER_VALUE = -2.75
def calculate_distance(distances):
    distances += 0.0000001
    #print GLOBAL_POWER_VALUE
    return distances ** GLOBAL_POWER_VALUE#-2.75

def remove_infrequent_places(feature, label, th=5):
    places, idx, counts = np.unique(label, return_inverse=True, 
                                                 return_counts=True)
    count_per_row = counts[idx]
    feature = feature[count_per_row >= th]
    label = label[count_per_row >= th]
    return feature, label

def get_data(filename, is_train_data):
    NUM_FEATURES = 14
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        
        record_array = np.empty([num_entires, 4], dtype="f8")
        feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
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
        
        feature[:,0] = (record_array[:,0]/10000.0)
        feature[:,1] = (record_array[:,1]/10000.0)
        feature[:,2] = np.log10(record_array[:,2])

        time = record_array[:,3]
        #minute = 2*np.pi*((time//5)%288)/288
        minute = 2*np.pi*((time)%1440)/1440
        feature[:,3] = (np.sin(minute)+1).round(4)
        feature[:,4] = (np.cos(minute)+1).round(4)
        
        day = 2*np.pi*((time//1440)%365)/365
        feature[:,5] = (np.sin(day)+1).round(4)
        feature[:,6] = (np.cos(day)+1).round(4)

        #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
        #feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001
        feature[:,7] = (time//525600)*1.0#*0.1#*0.00001
        #print feature[:,9]

        weekday = 2*np.pi*((time//1440)%7)/7
        feature[:,8] = (np.sin(weekday)+1).round(4)
        feature[:,9] = (np.cos(weekday)+1).round(4)

        accuracy = record_array[:,2]
        mask_0_45 = accuracy<45
        mask_45_100 = (accuracy >= 45) & (accuracy < 100)
        mask_100_200 = (accuracy >= 100) & (accuracy < 200)
        mask_200_1033 = (accuracy >= 200)
        feature[mask_0_45,10] = (accuracy[mask_0_45]/45)
        feature[mask_45_100,11] = (accuracy[mask_45_100]-45)/55
        feature[mask_100_200,12] = (accuracy[mask_100_200]-100)/100
        feature[mask_200_1033,13] = (accuracy[mask_200_1033]-200)/1024

    if(is_train_data):
        return (feature[:row_id], label[:row_id])
    else:
        return feature

def scale_features_onlyNN(feature, scale_factor):
    # location features
    feature[:,0] *= scale_factor[0]
    feature[:,1] *= scale_factor[1]
    #Accuracy log
    feature[:,2] *= scale_factor[2]
    #Time features
    feature[:,3] *= scale_factor[3]
    feature[:,4] *= scale_factor[3]
    feature[:,5] *= scale_factor[4]
    feature[:,6] *= scale_factor[4]
    feature[:,7] *= scale_factor[5]
    feature[:,8] *= scale_factor[6]
    feature[:,9] *= scale_factor[6]
    #Accuracy
    feature[:,10] *= scale_factor[7]
    feature[:,11] *= scale_factor[8]
    feature[:,12] *= scale_factor[9]
    feature[:,13] *= scale_factor[10]
    return feature

def scale_nn_feature(feature, scale_factor):
    feature[:,0] *= scale_factor[0]
    feature[:,1] *= scale_factor[1]
    return feature


def multi_model_prediction(nn0=49.0819, nn1=65.8511, nn=0.0, gh=0.0, gw=0.0, ga=0.0, cte=8.4592, gpv=-2.8465):
    eval_file = ct.XY_SD_SP_EVAL_TS_2SD_PATH+"5_39.csv"
    train_file = ct.XY_SD_SP_TR_TS_2SD_PATH+"5_39.csv"

    start_time = time.time()
    trf, trl = get_data(train_file, True)
    evf, evl = get_data(eval_file, True)
    end_time = time.time() - start_time


    global GLOBAL_POWER_VALUE 
    GLOBAL_POWER_VALUE = gpv

    trf = scale_nn_feature(trf, [nn0, nn1])
    evf = scale_nn_feature(evf, [nn0, nn1])
    n_neighbors = int((trl.size ** 0.5) / cte)
    knc = KNeighborsClassifier(n_neighbors=n_neighbors, weights=calculate_distance, p=1, n_jobs=-1, leaf_size=15)
    knc.fit(trf[:,0:2], trl)
    predict_knc = knc.predict_proba(evf[:,0:2])
    classes = knc.classes_

    gnb = GaussianNB()
    gnb.fit(trf[:,2:4], trl)
    predict_hours = gnb.predict_proba(evf[:,2:4])
    if (knc.classes_ != gnb.classes_).all():
        print "The Classes Do Not MATCH"

    gnb = GaussianNB()
    gnb.fit(trf[:,8:10], trl)
    predict_weekday = gnb.predict_proba(evf[:,8:10])

    gnb = GaussianNB()
    gnb.fit(trf[:,10:14], trl)
    predict_accuracy = gnb.predict_proba(evf[:,10:14])

    predict = nn*predict_knc + gh*predict_hours + gw*predict_weekday + ga*predict_accuracy

    knc_n = get_pred_actual_position(predict, classes, evf, evl)
    mean = knc_n[1]/np.sum(knc_n); score = (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])/np.sum(knc_n)
    # print "sf", sf, 
    # print "top1,2,3", knc_n[1], knc_n[2], np.sum(knc_n[1:4]), "top-1,10,20,30", knc_n[0], np.sum(knc_n[1:10]), np.sum(knc_n[1:20]), np.sum(knc_n[1:30])
    # print "mean", mean, "score", score

    return score


# sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.5138, sf4=0.077439, sf5=0.310318, sf6=0.058137, sf7=0.1371, sf8=0.1, sf9=0.1, cte=7.65 0.57738
# {'max_val': 0.58031664527171589, 'max_params': {'sf6': 0.46963133367621462, 'sf7': 0.158978618398706, 'sf5': 0.18514406935149511, 'sf3': 0.43798236154327763}}
# sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.43798, sf4=0.077439, sf5=0.18514, sf6=0.46963, sf7=0.15897, sf8=0.1, sf9=0.1, cte=7.65 0.580316645272
# sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.4655, sf4=0.077439, sf5=0.18514, sf6=0.46963, sf7=0.15897, sf8=0.1, sf9=0.1, cte=8.4592 - 0.580864356012
# sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.4655, sf4=0.18514, sf5=0.46963, sf6=0.15897, cte=8.4592 0.580864356012
# sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.4655, sf4=0.18514, sf5=0.46963, sf6=0.15897, cte=8.4592, gpv=-2.8465  0.581032

#Optimising accuracy
# {'max_val': 0.58152332049636291, 'max_params': {'sf10': 0.46073844421128463, 'sf7': 0.49775812013822063, 'sf2': 0.55002881263155412, 'sf8': 0.18975171704703428, 'sf9': 0.09772323821239208}}
# {'max_val': 0.58139780345171865, 'max_params': {'sf10': 0.68572474786529913, 'sf7': 0.82321913548590264, 'sf8': 0.29113748193252775, 'sf9': 0.15927110940332492}}

def do_prediction(sf0=17.9548, sf1= 38.7297, sf2=0.0, sf3=0.4655, sf4=0.18514, sf5=0.46963, sf6=0.15897, sf7=0.49775, sf8=0.18975, sf9=0.09772, sf10=0.46073, cte=8.4592, gpv=-2.8465):
    eval_file = ct.XY_SD_SP_EVAL_TS_2SD_PATH+"5_39.csv"
    train_file = ct.XY_SD_SP_TR_TS_2SD_PATH+"5_39.csv"

    #sf = [0.6, 0.32535, 0.56515, 0.2670, 22, 52, 0.51985] #[accuracy, day_of_year_sin/cos, minute_sin/cos, weekday_sin/cos, x, y, year]
    sf = [22, 52, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
    #sf = [1.0, 1.0, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
    sf = [sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9, sf10]
    global GLOBAL_POWER_VALUE 
    GLOBAL_POWER_VALUE = gpv

    start_time = time.time()
    trf, trl = get_data(train_file, True)
    evf, evl = get_data(eval_file, True)
    trf = scale_features_onlyNN(trf, sf)
    evf = scale_features_onlyNN(evf, sf)
    end_time = time.time() - start_time
    #print "time to load data ", end_time

    #trf, trl = remove_infrequent_places(trf, trl, 20)

    #cte = 7.65 #5.8
    n_neighbors = int((trl.size ** 0.5) / cte)
    #print n_neighbors, trl.size
    knc = KNeighborsClassifier(n_neighbors=n_neighbors,weights=calculate_distance, p=1, n_jobs=-1, leaf_size=15)
    #knc = GaussianNB()
    knc.fit(trf, trl)
    predict_knc = knc.predict_proba(evf)
    knc_n = get_pred_actual_position(predict_knc, knc.classes_, evf, evl)
    mean = knc_n[1]/np.sum(knc_n); score = (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])/np.sum(knc_n)
    # print "sf", sf, 
    # print "top1,2,3", knc_n[1], knc_n[2], np.sum(knc_n[1:4]), "top-1,10,20,30", knc_n[0], np.sum(knc_n[1:10]), np.sum(knc_n[1:20]), np.sum(knc_n[1:30])
    # print "mean", mean, "score", score
    return score

#cte 8.4592 0.54901
#cte 8.4523 gpv -1 0.54914
def do_multi_file_pred(sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.4655, sf4=0.18514, sf5=0.46963, sf6=0.15897, cte=8.4592, gpv=-2.8465):
    #(basename_train, basename_test, basename_pred, x_rng, y_rng) = (ct.XY_SD_SP_FUTR_PATH, ct.XY_SD_SP_TEST_PATH, ct.XY_SD_SP_TEST_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    (basename_train, basename_test, basename_pred, x_rng, y_rng) = (ct.XY_SD_SP_TR_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1
    #print num_x, num_y
    total_top_sum = 0.0; total_top3_sum = 0.0; total_pred = 0.0
    for idx_x in range(5,7): #1,num_x):
        for idx_y in range(39,42):#1,num_y):
            #print idx_x, idx_y
            sf = [sf0, sf1, sf2, sf3, sf4, sf5, sf6]
            train_file = basename_train+str(idx_x)+"_"+str(idx_y)+".csv"
            eval_file = basename_test+str(idx_x)+"_"+str(idx_y)+".csv"
            start_time = time.time()
            trf, trl = get_data(train_file, True, sf)
            evf, evl = get_data(eval_file, True, sf)
            #evf = get_data(eval_file, False, sf)
            end_time = time.time() - start_time

            n_neighbors = int((trl.size ** 0.5) / cte)
            knc = KNeighborsClassifier(n_neighbors=n_neighbors,weights=calculate_distance, p=1, n_jobs=-1, leaf_size=15)
            # le = preprocessing.LabelEncoder()
            # le.fit(trl)
            # print trl.shape, evf.shape, le.classes_.shape
            knc.fit(trf, trl)
            predict_knc = knc.predict_proba(feature)
            knc_n = get_pred_actual_position(predict_knc, evf, evl)
            total_pred += np.sum(knc_n); total_top_sum += knc_n[1]; total_top3_sum += (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])
            #print "mean", total_top_sum/total_pred, "score", total_top3_sum/total_pred
    return total_top3_sum/total_pred

def do_multi_file_pred_store_stats(sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.4655, sf4=0.077439, sf5=0.18514, sf6=0.46963, sf7=0.15897, sf8=0.1, sf9=0.1, cte=8.4592):
    #(basename_train, basename_test, basename_pred, x_rng, y_rng) = (ct.XY_SD_SP_FUTR_PATH, ct.XY_SD_SP_TEST_PATH, ct.XY_SD_SP_TEST_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    (basename_train, basename_test, basename_pred, x_rng, y_rng) = (ct.XY_SD_SP_TR_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1
    #print num_x, num_y
    total_top_sum = 0.0; total_top3_sum = 0.0; total_pred = 0.0
    for idx_x in range(num_x):
        for idx_y in range(num_y):
            #print idx_x, idx_y
            sf = [sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9]
            train_file = basename_train+str(idx_x)+"_"+str(idx_y)+".csv"
            eval_file = basename_test+str(idx_x)+"_"+str(idx_y)+".csv"
            start_time = time.time()
            trf, trl = get_data(train_file, True, sf)
            evf, evl = get_data(eval_file, True, sf)
            #evf = get_data(eval_file, False, sf)
            end_time = time.time() - start_time

            n_neighbors = int((trl.size ** 0.5) / cte)
            knc = KNeighborsClassifier(n_neighbors=n_neighbors,weights=calculate_distance, p=1, n_jobs=-1, leaf_size=15)
            # le = preprocessing.LabelEncoder()
            # le.fit(trl)
            # print trl.shape, evf.shape, le.classes_.shape
            knc.fit(trf, trl)
            predict_knc = knc.predict_proba(feature)
            knc_n = get_pred_actual_position(predict_knc, evf, evl)
            total_pred += np.sum(knc_n); total_top_sum += knc_n[1]; total_top3_sum += (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])
            mean = knc_n[1]/np.sum(knc_n); score = (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])/np.sum(knc_n)
            print "idx_x", idx_x, "idx_y", idx_y,
            print "top1,2,3", knc_n[1], knc_n[2], np.sum(knc_n[1:4]), "top-1,10,20,30", knc_n[0], np.sum(knc_n[1:10]), np.sum(knc_n[1:20]), np.sum(knc_n[1:30]),
            print "mean", mean, "score", score,
            print "total_mean", total_top_sum/total_pred, "total_score", total_top3_sum/total_pred
    return total_top3_sum/total_pred

def do_bayesian_multiple_file_opt():
    value_ranges = {'cte':(0,10), 'gpv':(-1,-3)}
    bo = BayesianOptimization(do_multi_file_pred, value_ranges)
    bo.maximize(init_points=10, n_iter=100, kappa=3.29)
    print bo.res['max']


def do_bayesian_single_file_opt():
    #value_ranges = {'sf0':(0, 100), 'sf1':(0, 100)}
    #value_ranges = {'sf3':(0,1), 'sf4':(0,1), 'sf5':(0,1), 'sf6':(0,1)}
    #value_ranges = {'sf7':(0,1), 'sf3':(0,1), 'sf5':(0,1), 'sf6':(0,1)}
    #value_ranges = {'sf3':(0,1)}
    #value_ranges = {'gpv':(-2,-3)}
    value_ranges = {'sf7':(0,1), 'sf8':(0,1), 'sf9':(0,1), 'sf10':(0,1)}
    bo = BayesianOptimization(do_prediction, value_ranges)
    bo.maximize(init_points=10, n_iter=100, kappa=3.29)
    print bo.res['max']

#{'max_val': 0.41275994865211812, 'max_params': {'nn1': 65.851142920123749, 'nn0': 49.081906354232657}}
def do_bay_opt_multi_model(): # multi_model_prediction(nn0=0.0, nn1=0.0, nn=0.0, gh=0.0, gw=0.0, ga=0.0)
    #value_ranges = {'nn0':(0,100), 'nn1':(0,100), 'nn':(0,1), 'gh':(0,1), 'gw':(0,1), 'ga':(0,1)}
    value_ranges = {'nn':(0,1), 'gh':(0,1)}
    bo = BayesianOptimization(multi_model_prediction, value_ranges)
    bo.maximize(init_points=10, n_iter=100, kappa=3.29)
    print bo.res['max']

# Using Gaussian Model
# https://www.kaggle.com/beyondbeneath/facebook-v-predicting-check-ins/testing
# https://www.kaggle.com/c/facebook-v-predicting-check-ins/forums/t/21845/accuracy-and-std-dev/124851
# https://www.kaggle.com/nigelhenry/facebook-v-predicting-check-ins/accuracy-explained/output
def do_gaussian_model_based_pred(train_file, eval_file):
    print "Start"

if __name__ == "__main__":
    # do_bayesian_multiple_file_opt()
    # do_bayesian_single_file_opt()
    do_bay_opt_multi_model()

    # print do_prediction()
    # print do_multi_file_pred(cte=8.4592), do_multi_file_pred(cte=7.65)
    # multi_model_prediction()
    #do_multi_file_pred()
    #do_multi_file_pred_store_stats()
# Yesterday's benchmark
# top1,2,3 3369.0 838.0 4598.0 top-1,10,20,30 539.0 5362.0 5608.0 5688.0
# mean 0.48053059478 score 0.558697760662

# Starting Bench mark
# top1,2,3 3360.0 855.0 4615.0 top-1,10,20,30 539.0 5365.0 5600.0 5676.0
# mean 0.479246897732 score 0.559050064185

# Using distance scaled by 2.75
# top1,2,3 3401.0 852.0 4622.0 top-1,10,20,30 539.0 5346.0 5578.0 5624.0
# mean 0.485094850949 score 0.563224932249

#   sf = [22, 52, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
# top1,2,3 3424.0 864.0 4684.0 top-1,10,20,30 539.0 5367.0 5568.0 5597.0
# mean 0.48837541007 score 0.568632149479


# month = 2*np.pi*((time//1440)%30)/30
# feature[:,5] = scale_factor[4]*(np.sin(month)+1).round(4)*0.0
# feature[:,6] = scale_factor[4]*(np.cos(month)+1).round(4)*0.0
# Month feature gave a best of zero in BayesianOptimisation


#Optimising accuracy
# {'max_val': 0.58152332049636291, 'max_params': {'sf10': 0.46073844421128463, 'sf7': 0.49775812013822063, 'sf2': 0.55002881263155412, 'sf8': 0.18975171704703428, 'sf9': 0.09772323821239208}}
# {'max_val': 0.58139780345171865, 'max_params': {'sf10': 0.68572474786529913, 'sf7': 0.82321913548590264, 'sf8': 0.29113748193252775, 'sf9': 0.15927110940332492}}
# def get_data(filename, is_train_data):
#     NUM_FEATURES = 14
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         record_array = np.empty([num_entires, 4], dtype="f8")
#         feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             record_array[row_id] = data_element[1:5]
#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
        
#         feature[:,0] = (record_array[:,0]/10000.0)
#         feature[:,1] = (record_array[:,1]/10000.0)
#         feature[:,2] = np.log10(record_array[:,2])

#         time = record_array[:,3]
#         #minute = 2*np.pi*((time//5)%288)/288
#         minute = 2*np.pi*((time)%1440)/1440
#         feature[:,3] = (np.sin(minute)+1).round(4)
#         feature[:,4] = (np.cos(minute)+1).round(4)
        
#         day = 2*np.pi*((time//1440)%365)/365
#         feature[:,5] = (np.sin(day)+1).round(4)
#         feature[:,6] = (np.cos(day)+1).round(4)

#         #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#         #feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001
#         feature[:,7] = (time//525600)*1.0#*0.1#*0.00001
#         #print feature[:,9]

#         weekday = 2*np.pi*((time//1440)%7)/7
#         feature[:,8] = (np.sin(weekday)+1).round(4)
#         feature[:,9] = (np.cos(weekday)+1).round(4)

#         accuracy = record_array[:,2]
#         mask_0_45 = accuracy<45
#         mask_45_100 = (accuracy >= 45) & (accuracy < 100)
#         mask_100_200 = (accuracy >= 100) & (accuracy < 200)
#         mask_200_1033 = (accuracy >= 200)
#         feature[mask_0_45,10] = (accuracy[mask_0_45]/45)
#         feature[mask_45_100,11] = (accuracy[mask_45_100]-45)/55
#         feature[mask_100_200,12] = (accuracy[mask_100_200]-100)/100
#         feature[mask_200_1033,13] = (accuracy[mask_200_1033]-200)/1024

#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature

# def scale_features_onlyNN(feature, scale_factor):
#     # location features
#     feature[:,0] *= scale_factor[0]
#     feature[:,1] *= scale_factor[1]
#     #Accuracy log
#     feature[:,2] *= scale_factor[2]
#     #Time features
#     feature[:,3] *= scale_factor[3]
#     feature[:,4] *= scale_factor[3]
#     feature[:,5] *= scale_factor[4]
#     feature[:,6] *= scale_factor[4]
#     feature[:,7] *= scale_factor[5]
#     feature[:,8] *= scale_factor[6]
#     feature[:,9] *= scale_factor[6]
#     #Accuracy
#     feature[:,10] *= scale_factor[7]
#     feature[:,11] *= scale_factor[8]
#     feature[:,12] *= scale_factor[9]
#     feature[:,13] *= scale_factor[10]
#     return feature


# def scale_features_onlyNN(feature, scale_factor):
#     feature[:,0] *= scale_factor[0]
#     feature[:,1] *= scale_factor[1]
#     feature[:,2] *= scale_factor[2]
#     feature[:,3] *= scale_factor[3]
#     feature[:,4] *= scale_factor[3]
#     feature[:,5] *= scale_factor[4]
#     feature[:,6] *= scale_factor[4]
#     feature[:,7] *= scale_factor[5]
#     feature[:,8] *= scale_factor[6]
#     feature[:,9] *= scale_factor[6]
#     return feature
# def get_data(filename, is_train_data):
#     NUM_FEATURES = 10
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         record_array = np.empty([num_entires, 4], dtype="f8")
#         feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             record_array[row_id] = data_element[1:5]
#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
        
#         feature[:,0] = (record_array[:,0]/10000.0)
#         feature[:,1] = (record_array[:,1]/10000.0)
#         feature[:,2] = np.log10(record_array[:,2])

#         time = record_array[:,3]
#         #minute = 2*np.pi*((time//5)%288)/288
#         minute = 2*np.pi*((time)%1440)/1440
#         feature[:,3] = (np.sin(minute)+1).round(4)
#         feature[:,4] = (np.cos(minute)+1).round(4)
        
#         day = 2*np.pi*((time//1440)%365)/365
#         feature[:,5] = (np.sin(day)+1).round(4)
#         feature[:,6] = (np.cos(day)+1).round(4)

#         #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#         #feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001
#         feature[:,7] = (time//525600)*1.0#*0.1#*0.00001
#         #print feature[:,9]

#         weekday = 2*np.pi*((time//1440)%7)/7
#         feature[:,8] = (np.sin(weekday)+1).round(4)
#         feature[:,9] = (np.cos(weekday)+1).round(4)

#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature



#sf0=17.954844860455644, sf1= 38.729716566060198, sf2=0.6, sf3=0.32535, sf4=0.0056515, sf5=0.2670, sf6=0.051985, cte=7.65 Value:  0.57150620453572953
#sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.32535, sf4=0.0056515, sf5=0.2670, sf6=0.051985, cte=7.65 Value: 0.57188988732
#sf0=17.9548, sf1= 38.7297, sf2=0.6362, sf3=0.5138, sf4=0.077439, sf5=0.310318, sf6=0.058137, cte=7.65 Value: 0.5727
#{'max_val': 0.57129796034802449, 'max_params': {'sf6': 0.089621891084420646, 'sf4': 0.065331465008290499, 'sf5': 0.34658546377016275, 'sf2': 0.57456520568176528, 'sf3': 0.66158385152780785, 'sf0': 33.778233953658408, 'sf1': 60.453131360325003}}
#{'max_val': 0.5725531307944659, 'max_params': {'sf7': 1.1540925213842166, 'sf8': 0.39398528120372223, 'sf9': 0.21719385956725357}}
# def get_data(filename, is_train_data, scale_factor):
#     NUM_FEATURES = 13
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         record_array = np.empty([num_entires, 4], dtype="f8")
#         feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             record_array[row_id] = data_element[1:5]
#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
        
#         feature[:,0] = scale_factor[0]*(record_array[:,0]/10000.0)
#         feature[:,1] = scale_factor[1]*(record_array[:,1]/10000.0)
#         feature[:,2] = scale_factor[2]*np.log10(record_array[:,2])*0.0

#         time = record_array[:,3]
#         minute = 2*np.pi*((time)%1440)/1440
#         feature[:,3] = scale_factor[3]*(np.sin(minute)+1).round(4)
#         feature[:,4] = scale_factor[3]*(np.cos(minute)+1).round(4)

#         month = 2*np.pi*((time//1440)%30)/30
#         feature[:,5] = scale_factor[4]*(np.sin(month)+1).round(4)#*0.01
#         feature[:,6] = scale_factor[4]*(np.cos(month)+1).round(4)#*0.01
        
#         day = 2*np.pi*((time//1440)%365)/365
#         weekday = 2*np.pi*((time//1440)%7)/7
#         feature[:,7] = scale_factor[5]*(np.sin(day)+1).round(4)
#         feature[:,8] = scale_factor[5]*(np.cos(day)+1).round(4)

#         #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#         feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001

#         accuracy = record_array[:,2]
#         mask_0_45 = accuracy<45
#         mask_45_125 = (accuracy >= 45) & (accuracy < 125)
#         mask_125_1033 = accuracy >= 125
#         feature[mask_0_45,10] = scale_factor[7]*(accuracy[mask_0_45]/45)
#         feature[mask_45_125,11] = scale_factor[8]*((accuracy[mask_45_125]-45)/80)
#         feature[mask_125_1033,12] = scale_factor[9]*((accuracy[mask_125_1033]-125)/160)
#         #print feature[:10,10:]

#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature


# benchmark after matching with kaggle scripts
# top1,2,3 3424.0 864.0 4684.0 top-1,10,20,30 539.0 5367.0 5568.0 5597.0
# mean 0.48837541007 score 0.568632149479
# The value on leader board:  0.56789
# sf = [22, 52, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
# def calculate_distance(distances):
#     return distances ** -2.75 # -2.22
# def get_data(filename, is_train_data, scale_factor):
#     NUM_FEATURES = 10
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             feature[row_id][0] = scale_factor[0]*(data_element[1]/10000.0)
#             feature[row_id][1] = scale_factor[1]*(data_element[2]/10000.0)
#             feature[row_id][2] = scale_factor[2]*np.log10(data_element[3])*1.0#/1033.0 #np.log()*100

#             minute = 2*np.pi*((data_element[4])%1440)/1440
#             feature[row_id][3] = scale_factor[3]*(np.sin(minute)+1).round(4)
#             feature[row_id][4] = scale_factor[3]*(np.cos(minute)+1).round(4)

#             month = 2*np.pi*((data_element[4]//1440)%30)/30
#             feature[row_id][5] = scale_factor[4]*(np.sin(month)+1).round(4)#*0.01
#             feature[row_id][6] = scale_factor[4]*(np.cos(month)+1).round(4)#*0.01
            
#             day = 2*np.pi*((data_element[4]//1440)%365)/365
#             weekday = 2*np.pi*((data_element[4]//1440)%7)/7
#             feature[row_id][7] = scale_factor[5]*(np.sin(day)+1).round(4)
#             feature[row_id][8] = scale_factor[5]*(np.cos(day)+1).round(4)

#             #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#             feature[row_id][9] = scale_factor[6]*(data_element[4]//(30*1440))#*0.1#*0.00001

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#       return (feature[:row_id], label[:row_id])
#     else:
#         return feature





# To be used when experimenting with optimising parameters for accuracy based on regions
# def get_data(filename, is_train_data, scale_factor):
#     NUM_FEATURES = 10
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             feature[row_id][0] = scale_factor[0]*(data_element[1]/10000.0)
#             feature[row_id][1] = scale_factor[1]*(data_element[2]/10000.0)
#             feature[row_id][2] = scale_factor[2]*np.log10(data_element[3])*1.0#/1033.0 #np.log()*100

#             minute = 2*np.pi*((data_element[4])%1440)/1440
#             feature[row_id][3] = scale_factor[3]*(np.sin(minute)+1).round(4)
#             feature[row_id][4] = scale_factor[3]*(np.cos(minute)+1).round(4)

#             month = 2*np.pi*((data_element[4]//1440)%30)/30
#             feature[row_id][5] = scale_factor[4]*(np.sin(month)+1).round(4)#*0.01
#             feature[row_id][6] = scale_factor[4]*(np.cos(month)+1).round(4)#*0.01
            
#             day = 2*np.pi*((data_element[4]//1440)%365)/365
#             weekday = 2*np.pi*((data_element[4]//1440)%7)/7
#             feature[row_id][7] = scale_factor[5]*(np.sin(day)+1).round(4)
#             feature[row_id][8] = scale_factor[5]*(np.cos(day)+1).round(4)

#             #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#             feature[row_id][9] = scale_factor[6]*(data_element[4]//(30*1440))#*0.1#*0.00001

#             # accuracy = data_element[3]
#             # if(accuracy < 45):
#             #     feature[row_id][10] = accuracy*0.0#*0.0085
#             #     feature[row_id][11] = 0.0
#             #     feature[row_id][12] = 0.0
#             # elif(accuracy < 125):
#             #     feature[row_id][10] = 0.0
#             #     feature[row_id][11] = (accuracy)*0.0#*0.005#0.01#0.005
#             #     feature[row_id][12] = 0.0
#             # else:
#             #     feature[row_id][10] = 0.0
#             #     feature[row_id][11] = 0.0
#             #     feature[row_id][12] = (accuracy-124)*0.0

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#       return (feature[:row_id], label[:row_id])
#     else:
#         return feature


# Not matched with kaggle scripts but used t give a very stable prediction
# top1,2,3 3360.0 855.0 4615.0 top-1,10,20,30 539.0 5365.0 5600.0 5676.0
# mean 0.479246897732 score 0.559050064185
# def get_data_working(filename, is_train_data, offset_factor):
#     NUM_FEATURES = 8
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [int(float(x)) for x in data_element]
#             feature[row_id][0] = data_element[1]#/100000.0
#             feature[row_id][1] = data_element[2]#/100000.0
#             feature[row_id][2] = np.log2(data_element[3])*64 #/1033.0 #np.log()*100
#             #feature[row_id][3] = (data_element[4]%(60*24))
#             mod_value = (60*24.0); offset = offset_factor*mod_value
#             time_offset = ((data_element[4]+offset) % mod_value)/mod_value
#             feature[row_id][3] = (np.sin(2*np.pi*time_offset)+1)*mod_value/12 #10,11,13
#             feature[row_id][4] = (np.cos(2*np.pi*time_offset)+1)*mod_value/12
#             num_days = (data_element[4]//mod_value)
#             weekday_offset = (num_days%7)/7
#             feature[row_id][5] = (np.sin(2*np.pi*weekday_offset)+1)*13
#             feature[row_id][6] = (np.cos(2*np.pi*weekday_offset)+1)*13
#               #(num_days%7)*27 #25,30 # There seems to be a influence by the number of weeks
#             # yearday_offset = (num_days%365)/365
#             # feature[row_id][7] = (np.sin(2*np.pi*yearday_offset)+1)*100
#             # feature[row_id][8] = (np.cos(2*np.pi*yearday_offset)+1)*100
#             feature[row_id][7] = (num_days//3) #2,4,5 # This feature gives importance to the placeid which are logged in recently
#             # feature[row_id][7] = (num_days%30)*0.5 # There seems to be no influence from month
#             # Since nearest neighbor seems to depend on distance, scale the inputs to meatch the dimension or importance of each feature
#             # If adding a feature does not change the output metric then the feature is not given enough importance to have any significant impact
#             # So try out values in the order of 1,10,100 till they start having impact. Once they start having impact, fine tune it by increasing or decreasing
#             # to find out whether the output metric (mean) increases or decreases proportionaly or inversely. Then fine tune in that direction to get the apporpriate weights

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#       return (feature[:row_id], label[:row_id])
#     else:
#         return feature






# Not used

# def get_data_1(filename, is_train_data, scale_factor):
#     NUM_FEATURES = 10
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             feature[row_id][0] = scale_factor[0]*(data_element[1]/10000.0)
#             feature[row_id][1] = scale_factor[1]*(data_element[2]/10000.0)
#             feature[row_id][2] = scale_factor[2]*np.log10(data_element[3])#/1033.0 #np.log()*100

#             minute = 2*np.pi*((data_element[4]//5)%288)/288
#             feature[row_id][3] = scale_factor[3]*(np.sin(minute)+1).round(4)
#             feature[row_id][4] = scale_factor[3]*(np.cos(minute)+1).round(4)

#             day = 2*np.pi*((data_element[4]//1440)%365)/365
#             feature[row_id][5] = scale_factor[4]*(np.sin(day)+1).round(4)
#             feature[row_id][6] = scale_factor[4]*(np.cos(day)+1).round(4)

#             weekday = 2*np.pi*((data_element[4]//1440)%7)/7
#             feature[row_id][7] = scale_factor[5]*(np.sin(day)+1).round(4)
#             feature[row_id][8] = scale_factor[5]*(np.cos(day)+1).round(4)

#             feature[row_id][9] = scale_factor[6]*(data_element[4]//525600)

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#       return (feature[:row_id], label[:row_id])
#     else:
#         return feature


# def get_data(filename, is_train_data, offset_factor=0.0):
#     NUM_FEATURES = 7
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='i4')
#         label = np.empty([num_entires, 1], dtype='i4').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [int(float(x)) for x in data_element]
#             feature[row_id][0] = data_element[1]#/100000.0
#             feature[row_id][1] = data_element[2]#/100000.0
#             feature[row_id][2] = np.log2(data_element[3])*64 #/1033.0 #np.log()*100
#             #feature[row_id][3] = (data_element[4]%(60*24))
#             mod_value = (60*24.0); offset = offset_factor*mod_value
#             time_offset = ((data_element[4]+offset) % mod_value)/mod_value
#             feature[row_id][3] = (np.sin(2*np.pi*time_offset)+1)*mod_value/12 #10,11,13
#             feature[row_id][4] = (np.cos(2*np.pi*time_offset)+1)*mod_value/12
#             num_days = (data_element[4]//mod_value)
#             feature[row_id][5] = (num_days%7)*27 #25,30 # There seems to be a influence by the number of weeks
#             feature[row_id][6] = (num_days//3) #2,4,5 # This feature gives importance to the placeid which are logged in recently
#             # feature[row_id][7] = (num_days%30)*0.5 # There seems to be no influence from month
#             # Since nearest neighbor seems to depend on distance, scale the inputs to meatch the dimension or importance of each feature
#             # If adding a feature does not change the output metric then the feature is not given enough importance to have any significant impact
#             # So try out values in the order of 1,10,100 till they start having impact. Once they start having impact, fine tune it by increasing or decreasing
#             # to find out whether the output metric (mean) increases or decreases proportionaly or inversely. Then fine tune in that direction to get the apporpriate weights

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature
# def calculate_distance(distances):
#     distances[distances < .0001] = .0001
#     return distances ** -2