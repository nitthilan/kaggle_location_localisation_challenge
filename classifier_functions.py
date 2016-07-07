#!/usr/bin/python
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets

from sklearn import svm
from sklearn import preprocessing
import time
from sklearn.externals import joblib
import constants as ct
import gzip
import utils as ut

def get_data(filename, is_train_data):
    NUM_FEATURES = 7
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        feature = np.empty([num_entires, NUM_FEATURES], dtype='i4')
        label = np.empty([num_entires, 1], dtype='i4').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [int(float(x)) for x in data_element]
            # if(unique_place_id_num and unique_place_id_num[data_element[5]] < 100):
            # 	continue
            feature[row_id][0] = data_element[1]
            feature[row_id][1] = data_element[2]
            feature[row_id][2] = np.log2(data_element[3])*100

            nH = int(data_element[4]/60.0) # Num Hours - Cannot be used as feature
            Minute = data_element[4]%60.0 # May not be useful
            HiD24 = nH%24 #Hour in Day (0-24)
            HiD18 = (nH+18)%24 #Hour in Day (12-12)
            HiD6 = (nH+6)%24 #Hour in Day (12-12)
            nD = int(nH/24) #numDays - Cannot be used as feature
            DiW = nD%7 #Day in Week
            DiM = nD%30.5 #Day in Month
            DiBW = nD%14 #Day in Bi-Week
            DiFW = nD%28 #Day in Four-Week
            DiY = nD%365 #Day in Year

            feature[row_id][3] = HiD24
            feature[row_id][4] = DiW
            feature[row_id][5] = DiM
            # feature[row_id][6] = int(DiBW/7)
            feature[row_id][6] = int(DiY/182.5)
            # feature[row_id][7] = int(DiM/7)
            # feature[row_id][8] = int(nD/365)
            # feature[row_id][6] = data_element[1] + data_element[3]
            # feature[row_id][6] = HiD6
            # feature[row_id][7] = HiD18
            if(is_train_data):
                label[row_id] = data_element[5]
            #print feature[row_id], label[row_id]
            row_id = row_id + 1
    #print feature.shape, label.shape
    print row_id, num_entires
    if(is_train_data):
        return (feature[:row_id], label[:row_id])
    else:
       return feature

def get_trained_classifier(train_filename, classifier):
	start_time = time.time()
	feature, label = get_data(train_filename, True)
	train_file_read_time = time.time() - start_time
	#print feature.shape, label.shape

	start_time = time.time()
	classifier.fit(feature, label)
	fit_time = time.time() - start_time
	print train_file_read_time, fit_time
	return classifier

def get_prediction(classifier, test_filename):
    start_time = time.time()
    feature = get_data(test_filename, False)
    test_file_read_time = time.time() - start_time

    start_time = time.time()
    predict = classifier.predict_proba(feature)
    predict_time = time.time() - start_time

    # argnmax is very slow compared to argmax. So better to replace value with negative value than Nan.
    start_time = time.time()
    top_0 = np.argmax(predict, axis=1)
    
    classes = classifier.classes_ 
    row_id = np.arange(len(predict))
    
    top_class_0 = classes[top_0]
    col_0 = predict[row_id, top_0]
    predict[row_id, top_0] = -1.0
    top_1 = np.argmax(predict, axis=1)
    top_class_1 = classes[top_1]
    col_1 = predict[row_id, top_1]
    predict[row_id, top_1] = -1.0
    top_2 = np.argmax(predict, axis=1)
    top_class_2 = classes[top_2]
    col_2 = predict[row_id, top_2]
    #argmax_top_n = np.vstack((top_0, top_1, top_2)).transpose()
    #argmax_predict_value = np.vstack((col_0, col_1, col_2)).transpose()
    top_n_predicted = np.vstack((top_class_0, col_0, top_class_1, col_1, top_class_2, col_2)).transpose()
    argmax_time = time.time() - start_time

    #print argmax_time, (argmax_predict_value == top_n).all(), (predict_value == argmax_predict_value).all()
    #print argmax_top_n[5], top_n[5], argmax_predict_value[5], predict_value[5]

    print test_file_read_time, predict_time, argmax_time #argsort_time, stack_array_time

    #return top_n, predict_value
    #return argmax_top_n, argmax_predict_value
    return top_n_predicted

def get_pred_actual_position(classifier, test_filename):
	start_time = time.time()
	feature, label = get_data(test_filename, True)
	test_file_read_time = time.time() - start_time

	start_time = time.time()
	predict = classifier.predict_proba(feature)
	predict_time = time.time() - start_time

	pred_pos = np.argsort(predict, axis=1, kind="heapsort")
	# print pred_pos.shape
	# print pred_pos[1:2, 60:70]
	# print predict[1:2, 60:70]

	# sorted_value = np.argsort(predict[1, :])
	# idx = np.where( sorted_value == 614 )
	# print  sorted_value[60:70], sorted_value, idx[0][0]

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

def get_top_n_prediction(classifier, test_filename):
	start_time = time.time()
	feature, label = get_data(test_filename, True)
	test_file_read_time = time.time() - start_time

	start_time = time.time()
	predict = classifier.predict_proba(feature)
	predict_time = time.time() - start_time

	pred_pos = np.argsort(predict, axis=1)

	classes = classifier.classes_
	inv_class_map = {}
	actual_idx = 0
	for idx in classes:
		inv_class_map[idx] = actual_idx;
		actual_idx += 1

	return pred_pos, classes





def dump_predicted_n(top_n_predicted, top_n_filename):
        start_time = time.time()
        #predicted_filename = PREDICTED_TEST_FILENAME+"_pv_"+str(train_idx)+"_"+str(test_idx)+".csv"
        np.savetxt(top_n_filename, top_n_predicted, delimiter=",", fmt="%d,%0.2f,%d,%0.2f,%d,%0.2f");
        #np.savetxt(predicted_filename, predicted_value, delimiter=",", fmt="%0.6f");
        dump_time = time.time() - start_time
        print dump_time

def do_prediction(train_filename, eval_filename, pred_filename):
	classifier = cf.get_trained_classifier(train_file)
	top_n_predicted = cf.get_prediction(classifier, eval_file)
	cf.dump_predicted_n(top_n_predicted, pred_file)
