from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
import time

#DR_FILENAME = "./data/distributed_record_local_label/drll0.csv"
DR_FILENAME = "./data/dr0.csv"


def get_train_data_float(filename):
	with open(DR_FILENAME) as infile:
		feature_array = []; label_array = []; row_id = 0
		for line in islice(infile,1,None):
			data_element = line.strip().split(",")
			data_element = [float(x) for x in data_element]
			feature_array.append(data_element[1:5])
			label_array.append(int(data_element[5]))
			row_id = row_id + 1
	return (np.array(feature_array), np.array(label_array))

def get_train_data(filename):
        with open(DR_FILENAME) as infile:
                num_entires = 0
                for line in islice(infile,1,None):
                        num_entires = num_entires + 1
                feature = np.empty([num_entires, 4], dtype='i4')
                label = np.empty([num_entires, 1], dtype='i4').ravel()
                row_id = 0
                infile.seek(0)
                for line in islice(infile,1,None):
                        data_element = line.strip().split(",")
                        data_element = [float(x) for x in data_element]
                        feature[row_id][0] = int(data_element[1]*65536)
                        feature[row_id][1] = int(data_element[2]*65536)
                        feature[row_id][2] = int(data_element[3]*1033)
                        feature[row_id][3] = int(data_element[4]*786239)
                        label[row_id] = int(data_element[5])
                        #print feature[row_id], label[row_id]
                        row_id = row_id + 1
                print feature.shape, label.shape
        return (feature, label)


def get_performance(np_feature, np_label, classifier):
	#print np_feature.shape, np_label.shape
	NUM_TRAIN = int(0.8*np_feature.shape[0])

	# weights has two options 'uniform', 'distance'
	classifier.fit(np_feature[:NUM_TRAIN], np_label[:NUM_TRAIN,])
	predict = classifier.predict(np_feature[NUM_TRAIN:])
	compare = predict==np_label[NUM_TRAIN:]
	return compare.mean()


def get_top_n(np_feature, np_label, classifier):
	#print np_feature.shape, np_label.shape
	NUM_TRAIN = int(0.8*np_feature.shape[0])

	# weights has two options 'uniform', 'distance'
	classifier.fit(np_feature[:NUM_TRAIN], np_label[:NUM_TRAIN,])
	predict = classifier.predict_proba(np_feature[NUM_TRAIN:])
	#http://stackoverflow.com/questions/28568034/getting-scikit-learn-randomforestclassifier-to-output-top-n-results
	n = 3
	top_n = np.argsort(predict)[:,:-n-1:-1]
	top_predict = classifier.predict(np_feature[NUM_TRAIN:])
	#print top_n.shape, predict.shape, top_n, top_predict
	#print top_n[:,0].shape, top_predict.shape
	top_compare = top_predict==np_label[NUM_TRAIN:]
	top_n_compare = top_n[:,0]==np_label[NUM_TRAIN:]
	print (top_n[:,0] == top_predict).all(), top_compare.mean(), top_n_compare.mean()
	return


(np_feature, np_label_without_pre_processing) = get_train_data(DR_FILENAME)
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
le = preprocessing.LabelEncoder()
np_label = le.fit_transform(np_label_without_pre_processing)
#print le.classes_


# Nearest neighbor algorithm: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.get_params
# Weight:
# Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute 
# more to the fit. This can be accomplished through the weights keyword. The default value, 
# weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights 
# proportional to the inverse of the distance from the query point. Alternatively, a user-defined function 
# of the distance can be supplied which is used to compute the weights.
# weight, num_neighbors, algorithm, mean_correct_value, total_time
# uniform 1 kd_tree 0.749281473195 1.55174517632
# uniform 3 kd_tree 0.722209792617 3.0338640213
# uniform 5 kd_tree 0.704603734618 3.45425486565
# uniform 10 kd_tree 0.665330006024 4.44531583786
# uniform 15 kd_tree 0.636812666724 5.49731707573
# uniform 20 kd_tree 0.613905860081 6.59826517105
# uniform 25 kd_tree 0.595077876258 7.83416604996
# distance 1 kd_tree 0.749281473195 1.64854598045
# distance 3 kd_tree 0.746321314861 3.14151501656
# distance 5 kd_tree 0.739041390586 4.65441203117
# distance 10 kd_tree 0.717287668875 6.50194501877
# distance 15 kd_tree 0.700645383358 8.29478502274
# distance 20 kd_tree 0.686223216591 9.91831898689
# distance 25 kd_tree 0.673539282334 12.5876750946
# for weight in ['uniform', 'distance']:
# 	for n_neighbors in [1,3,5,10,15,20,25]:
# 		for algorithm in [ 'kd_tree']:#'brute' takes a lot of time
# 			start_time = time.time()	
# 			classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weight, algorithm = algorithm)
# 			mean = get_performance(np_feature, np_label, classifier)
# 			total_time = time.time() - start_time
# 			print weight, n_neighbors, algorithm, mean, total_time

# Random forest classifier
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# num_estimators: number of trees in the forest
# num_estimators, mean_correct_value, total_time
# dr0
# 1 0.812735564926 8.86215400696
# 3 0.863264779279 27.1258850098
# 5 0.895516736942 47.5079510212
# 10 0.906376387574 106.926122904
# 15 0.91623784528 161.358649969
# 20 0.916685311075 240.075958014
# 25 0.916891833749 294.315912962
# dr1
# 1 0.844740057478 8.65076804161
# 3 0.872446608959 28.8303010464
# 5 0.888623104855 49.2628529072
# 10 0.903749849421 105.036150932
# 15 0.914436662135 157.265836954
# 20 0.91837753188 212.591512203
# 25 0.919358447056 290.361974955
# dr2
# 1 0.814358480176 9.26847600937
# 3 0.86720470815 29.4440858364
# 5 0.90002064978 51.8748180866
# 10 0.911894273128 111.555720091
# 15 0.916041437225 168.062342882
# 20 0.919448650881 235.267134905
# 25 0.92340652533 302.791250944
# Memeory seems to peak for num_estimator goes to 25. It seems to hit 20GB etc
# for n_estimators in [1,3,5,10,15,20,25]:
# 	start_time = time.time()	
# 	classifier = RandomForestClassifier(n_estimators, n_jobs=-1)
# 	mean = get_performance(np_feature, np_label, classifier)
# 	total_time = time.time() - start_time
# 	print n_estimators, mean, total_time

# start_time = time.time()	
# classifier = RandomForestClassifier(25, n_jobs=-1)
# mean = get_performance(np_feature, np_label, classifier)
# total_time = time.time() - start_time
# print mean, total_time






#start_time = time.time()	
#classifier = svm.SVC()
#mean = get_performance(np_feature, np_label, classifier)
#total_time = time.time() - start_time
#print n_estimators, mean, total_time



# NOTE: For nearest neighbor based on 'uniform' weights the prediction based on direct predict does not match the prediction based on predict_prob
# However it seems to match for weights based on 'distance'
# Same problem seems to be present for RandomForestClassifier
# for weight in ['uniform', 'distance']:
# 	for n_neighbors in [1,3,5,10,15,20,25]:
# 		for algorithm in [ 'kd_tree']:#'brute' takes a lot of time
# 			start_time = time.time()	
# 			classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weight, algorithm = algorithm)
# 			mean = get_top_n(np_feature, np_label, classifier)
# 			total_time = time.time() - start_time
# 			print weight, n_neighbors, algorithm, mean, total_time
# for n_estimators in [1,3,5,10,15,20,25]:
# 	start_time = time.time()	
# 	classifier = RandomForestClassifier(n_estimators)
# 	mean = get_top_n(np_feature, np_label, classifier)
# 	total_time = time.time() - start_time
# 	print n_estimators, mean, total_time

# Support Vector Machines to be tried out??

