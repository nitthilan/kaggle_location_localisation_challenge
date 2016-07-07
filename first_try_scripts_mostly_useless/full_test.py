#!/usr/bin/python
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
import time
from sklearn.externals import joblib

DRLL_FILENAME = "./data/distributed_record_local_label/drll"
TNS_FILENAME = "./data/test_normalised_split/tns"
PREDICTED_TEST_FILENAME = "./data/predicted_test_run1/pt"

NUM_TRAIN_FILE = 100
NUM_TEST_FILE = 30
NUM_ESTIMATOR = 15
ENABLE_TIME_FEATURE = True
START_IDX_TRAIN = 0

# for num_estimator of 10 and without time feature the random forrest consumes around 41.25Gb data
# for num_estimator of 5 and without time feature the random forrest consumes around 22.6Gb data

# performance for num estimator =10
# 2.16883301735 13.3984789848
#3.09706377983 16.8084769249 6.08643889427 0.0842809677124
#1.71282887459 13.190994978 5.54321599007 0.142879962921
#2.12693285942 11.9988811016 5.27968811989 0.0982921123505
#1.54462003708 11.9614899158 6.41844987869 0.151927947998
#2.78087806702 12.0450980663 5.99882388115 0.100534915924
#1.93086194992 12.5081288815 6.94426178932 0.0885500907898
#1.41800904274 12.8961420059 6.37180900574 0.14372086525
#1.60604190826 12.3869440556 6.58103990555 0.114083051682
#2.15900707245 11.5782868862 5.74196100235 0.0996360778809
#1.97678685188 12.04160285 6.0235350132 0.0995669364929
#2.10508203506 12.0709450245 5.39170479774 0.148865938187
#1.70566105843 11.8269860744 5.76385307312 0.100641012192
#2.18595004082 11.8868880272 5.47946405411 0.0889940261841
#1.66016507149 12.3426048756 5.7364590168 0.134126901627
#1.88557887077 11.0142300129 5.72241687775 0.0942049026489
#1.70017194748 11.4720420837 5.16381788254 0.11106801033
#1.77391886711 11.8965289593 5.2218811512 0.0941119194031
#1.29850506783 11.4833190441 6.06552696228 0.0958471298218
#1.64164996147 11.0249509811 5.60166692734 0.0966720581055
#2.17591714859 11.6120159626 5.37468695641 0.144608974457


def get_data(filename, is_train_data):
	with open(filename) as infile:
		num_entires = 0
		for line in islice(infile,1,None):
			num_entires = num_entires + 1
		if(ENABLE_TIME_FEATURE):
			feature = np.empty([num_entires, 4], dtype='i4')
		else:
			feature = np.empty([num_entires, 3], dtype='i4')

		if(is_train_data):		
			label = np.empty([num_entires, 1], dtype='i4').ravel()
		row_id = 0
		infile.seek(0)
		for line in islice(infile,1,None):
			data_element = line.strip().split(",")
			data_element = [float(x) for x in data_element]
			feature[row_id][0] = int(data_element[1]*10000)
			feature[row_id][1] = int(data_element[2]*10000)
			feature[row_id][2] = int(data_element[3]*1033)
			if(ENABLE_TIME_FEATURE):
				T = (data_element[3]*786239.0)/(24*60.0)
				D = int(T)
				H = int(24*(T-D))
				feature[row_id][3] = H
			if(is_train_data):
				label[row_id] = int(data_element[5])
			#print feature[row_id], label[row_id]
			row_id = row_id + 1
		
	if(is_train_data):
		#print feature.shape, label.shape
		return (feature, label)
	else:
		#print feature.shape
		return feature

def get_trained_classifier(fileidx):
	start_time = time.time()	
	dist_record_filename = DRLL_FILENAME+str(fileidx)+".csv"
	(feature, label) = get_data(dist_record_filename, True)
	train_file_read_time = time.time() - start_time
	
	start_time = time.time()
	classifier = RandomForestClassifier(NUM_ESTIMATOR, n_jobs=16)
	classifier.fit(feature, label)
	fit_time = time.time() - start_time
	print fileidx, train_file_read_time, fit_time
	return classifier

def get_prediction(classifier, fileidx):
	start_time = time.time()
	test_filename = TNS_FILENAME+str(fileidx)+".csv"
	feature = get_data(test_filename, False)
	test_file_read_time = time.time() - start_time

	# start_time = time.time()
	# classifier.predict(feature)
	# predict_top_time = time.time() - start_time

	start_time = time.time()
	predict = classifier.predict_proba(feature)
	predict_time = time.time() - start_time
	
	##http://stackoverflow.com/questions/28568034/getting-scikit-learn-randomforestclassifier-to-output-top-n-results
	#start_time = time.time()
	#n = 3
	#top_n = np.argsort(predict)[:,:-n-1:-1]
	#argsort_time = time.time() - start_time
	##http://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
	#start_time = time.time()
	#row_id = np.arange(len(predict))
	#col0 = predict[row_id, top_n[:,0]]
	#col1 = predict[row_id, top_n[:,1]]
	#col2 = predict[row_id, top_n[:,2]]
	##print top_n.shape, predict.shape, col0.shape, col1.shape
	#predict_value = np.vstack((col0, col1, col2)).transpose()
	##print predict_value.shape
	##print predict[5][top_n[5][0]], predict[5][top_n[5][1]], predict[5][top_n[5][2]], top_n[5], predict_value[5]
	##top_n_value = (top_n[0], predict[top_n[0]], top_n[1], predict[top_n[1]], top_n[2], predict[top_n[2]])
	#stack_array_time = time.time() - start_time

	# argnmax is very slow compared to argmax. So better to replace value with negative value than Nan.
	start_time = time.time()
	row_id = np.arange(len(predict))
	top_0 = np.argmax(predict, axis=1)
	col_0 = predict[row_id, top_0]
	predict[row_id, top_0] = -1.0
	top_1 = np.argmax(predict, axis=1)
	col_1 = predict[row_id, top_1]
	predict[row_id, top_1] = -1.0 
	top_2 = np.argmax(predict, axis=1)
	col_2 = predict[row_id, top_2]
	#argmax_top_n = np.vstack((top_0, top_1, top_2)).transpose()
	#argmax_predict_value = np.vstack((col_0, col_1, col_2)).transpose()
	top_n_predicted = np.vstack((top_0, col_0, top_1, col_1, top_2, col_2)).transpose()
	argmax_time = time.time() - start_time

	#print argmax_time, (argmax_predict_value == top_n).all(), (predict_value == argmax_predict_value).all()
	#print argmax_top_n[5], top_n[5], argmax_predict_value[5], predict_value[5]
	
	print fileidx, test_file_read_time, predict_time, argmax_time #argsort_time, stack_array_time

	#return top_n, predict_value
	#return argmax_top_n, argmax_predict_value
	return top_n_predicted

def dump_predicted_n(top_n_predicted, train_idx, test_idx):
	start_time = time.time()
	top_n_filename = PREDICTED_TEST_FILENAME+str(train_idx)+"_"+str(test_idx)+".csv"
	#predicted_filename = PREDICTED_TEST_FILENAME+"_pv_"+str(train_idx)+"_"+str(test_idx)+".csv"
	np.savetxt(top_n_filename, top_n_predicted, delimiter=",", fmt="%d,%0.5f,%d,%0.5f,%d,%0.5f");
	#np.savetxt(predicted_filename, predicted_value, delimiter=",", fmt="%0.6f");
	dump_time = time.time() - start_time
	print train_idx, test_idx, dump_time


def create_trained_model():
	for fileidx in range(START_IDX_TRAIN, NUM_TRAIN_FILE):
		classifier = get_trained_classifier(fileidx)
		for testfileidx in range(NUM_TEST_FILE):
			top_n_predicted = get_prediction(classifier, testfileidx)
			dump_predicted_n(top_n_predicted, fileidx, testfileidx)

if __name__ == "__main__":
	create_trained_model()

