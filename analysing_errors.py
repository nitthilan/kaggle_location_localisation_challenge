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


def get_unique_place_id(filename, start, end):
	unique_id = {}; 
	with open(filename) as infile:
		for line in islice(infile, start, end):
			data = [int(float(x)) for x in line.strip().split(",")]
			place_id = data[5]
			if(not unique_id.has_key(place_id)):
				unique_id[place_id] = []
			unique_id[place_id].append(data)
	unique_id_num = {}
	for key, value in unique_id.iteritems():
		unique_id_num[key] = len(value)
	return unique_id, unique_id_num

def get_missing_place_id(eval_place_id, train_place_id):
	missing_place_id = []; present_place_id = [];  num_missing_record = 0; num_present_record = 0
	for place_id, place_id_list in eval_place_id.iteritems():
		if(not train_place_id.has_key(place_id)):
			missing_place_id.append(place_id)
			num_missing_record = num_missing_record + len(place_id_list)
		else:
			present_place_id.append(place_id)
			num_present_record = num_present_record + len(place_id_list)
	return (missing_place_id, present_place_id, num_missing_record, num_present_record)

def get_error_records_list(eval_filename, top_n_predicted, train_place_id, pred_cost_position):
	zero_option = defaultdict(list); first_option = defaultdict(list); second_option = defaultdict(list); 
	not_in_list = defaultdict(list); not_in_train = defaultdict(list); not_in_list_pred = defaultdict(list);
	num_records_error = Counter()
	with open(eval_filename) as eval_file:
		idx = 0
		for eval_line in eval_file:
			#eval_line = eval_file.readline()
			eval_data = [int(float(x)) for x in eval_line.strip().split(",")]
			#pred = [int(float(x)) for x in pred_line.strip().split(",")]
			pred = top_n_predicted[idx]
			true_place_id = eval_data[5]; pred0 = pred[0]; pred1 = pred[2]; pred2 = pred[4]
			if(true_place_id == pred0):
				zero_option[true_place_id].append(eval_data)
				num_records_error["zero"] += 1
			elif (true_place_id == pred1):
				first_option[true_place_id].append(eval_data)
				num_records_error["one"] += 1
			elif (true_place_id == pred2):
				second_option[true_place_id].append(eval_data)
				num_records_error["two"] += 1 
			elif (train_place_id.has_key(true_place_id)):
				not_in_list[true_place_id].append(eval_data)
				not_in_list_pred[true_place_id].append((pred,pred_cost_position[idx]))
				num_records_error["none"] += 1
			else:
				not_in_train[true_place_id].append(eval_data)
				num_records_error["none_train"] += 1
			idx = idx + 1

			#if(eval_data[5] == pred[0])
	return (zero_option, first_option, second_option, not_in_list, not_in_train, not_in_list_pred, num_records_error)


def get_time_break_down(time_list):
	#print time_list
	nH = (time_list/60.0).astype(int) # Num Hours - Cannot be used as feature
	Minute = time_list%60.0
	HiD24 = nH%24 #Hour in Day (0-24)
	HiD6 = (nH+18)%24 #Hour in Day (12-12)
	nD = (nH/24).astype(int) #numDays - Cannot be used as feature
	DiW = nD%7 #Day in Week
	DiM = nD%30.5 #Day in Month
	DiBW = nD%14 #Day in Bi-Week
	DiFW = nD%28 #Day in Four-Week
	DiY = nD%365 #Day in Year
	return (HiD24.transpose(), DiW.transpose(), DiM.transpose(), HiD6.transpose())

def plot_graph(train_list, eval_list, plotfilename):
	#print eval_list
	# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
	# plt.axis((0,100000,0,100000))
	fig = plt.figure()

	fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.99, hspace=0.1, wspace=0.1)
	axarr = []; num_graph = 0
	for i in range(2):
		axarr.append([])
		for j in range(3):
			axarr[i].append(fig.add_subplot(2,3,num_graph+1))
			num_graph = num_graph + 1

	(HiD24_tr, DiW_tr, DiM_tr, HiD6_tr) = get_time_break_down(train_list[::, 4])
	(HiD24_ev, DiW_ev, DiM_ev, HiD6_ev) = get_time_break_down(eval_list[::, 4])
	# print train_list[::, 1].shape, HiD24_tr.shape
	# print eval_list[::, 1].shape, HiD24_ev.shape

	axarr[0][0].axis((0,786239,0,100000))
	axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="blue", alpha=0.5)
	axarr[0][0].scatter(eval_list[::, 4], eval_list[::, 1],  c="red", alpha=0.5)

	axarr[0][1].axis((0,786239,0,100000))
	axarr[0][1].scatter(train_list[::, 4], train_list[::, 2],  c="blue", alpha=0.5)
	axarr[0][1].scatter(eval_list[::, 4], eval_list[::, 2],  c="red", alpha=0.5)

	axarr[0][2].axis((0,786239,0,1033))
	axarr[0][2].scatter(train_list[::, 4], train_list[::, 3],  c="blue", alpha=0.5)
	axarr[0][2].scatter(eval_list[::, 4], eval_list[::, 3],  c="red", alpha=0.5)

	axarr[1][0].axis((0,100000,0,7))
	axarr[1][0].scatter(train_list[::, 2], DiW_tr,  c="blue", alpha=0.5)
	axarr[1][0].scatter(eval_list[::, 2], DiW_ev,  c="red", alpha=0.5)

	axarr[1][1].axis((0,100000,0,30))
	axarr[1][1].scatter(train_list[::, 2], DiM_tr,  c="blue", alpha=0.5)
	axarr[1][1].scatter(eval_list[::, 2], DiM_ev,  c="red", alpha=0.5)

	axarr[1][2].axis((0,100000,0,24))
	axarr[1][2].scatter(train_list[::, 2], HiD6_tr,  c="blue", alpha=0.5)
	axarr[1][2].scatter(eval_list[::, 2], HiD6_ev,  c="red", alpha=0.5)


	fig.savefig(plotfilename, format='pdf')
	return 

def plot_options(train_place_id, not_in_list, not_in_list_pred, place_id, plotfilename):
	fig = plt.figure()

	fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.99, hspace=0.1, wspace=0.1)
	axarr = []; num_graph = 0
	for i in range(2):
		axarr.append([])
		for j in range(3):
			axarr[i].append(fig.add_subplot(2,3,num_graph+1))
			num_graph = num_graph + 1


	print place_id
	print not_in_list_pred[place_id][0]
	print not_in_list_pred[place_id][10]
	start_time = 786239*2/3; end_time = 786239
	pred_list, pred_cost_pos = not_in_list_pred[place_id][0]
	error_list = not_in_list[place_id][0]
	error_list_full = np.array(not_in_list[place_id])
	axarr[0][0].axis((start_time,end_time,0,100000))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="yellow", alpha=0.5)
	axarr[0][0].scatter(error_list_full[::, 4], error_list_full[::, 1],  c="yellow", alpha=0.5)
	axarr[0][0].scatter(error_list[ 4], error_list[ 1],  c="red", alpha=1.0)

	axarr[0][1].axis((start_time,end_time,57500,62500))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[0][1].scatter(train_list[::, 4], train_list[::, 2],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[0][1].scatter(train_list[::, 4], train_list[::, 2],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[0][1].scatter(train_list[::, 4], train_list[::, 2],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[0][1].scatter(train_list[::, 4], train_list[::, 2],  c="yellow", alpha=0.5)
	axarr[0][1].scatter(error_list_full[::, 4], error_list_full[::, 2],  c="yellow", alpha=0.5)
	axarr[0][1].scatter(error_list[4], error_list[2],  c="red", alpha=1.0)

	axarr[0][2].axis((start_time,end_time,0,1033))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[0][2].scatter(train_list[::, 4], train_list[::, 3],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[0][2].scatter(train_list[::, 4], train_list[::, 3],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[0][2].scatter(train_list[::, 4], train_list[::, 3],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[0][2].scatter(train_list[::, 4], train_list[::, 3],  c="yellow", alpha=0.5)
	axarr[0][2].scatter(error_list_full[::, 4], error_list_full[::, 3],  c="yellow", alpha=0.5)
	axarr[0][2].scatter(error_list[4], error_list[3],  c="red", alpha=1.0)


	pred_list, pred_cost_pos = not_in_list_pred[place_id][10]
	error_list = not_in_list[place_id][10]
	axarr[1][0].axis((start_time,end_time,0,100000))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[1][0].scatter(train_list[::, 4], train_list[::, 1],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[1][0].scatter(train_list[::, 4], train_list[::, 1],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[1][0].scatter(train_list[::, 4], train_list[::, 1],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[1][0].scatter(train_list[::, 4], train_list[::, 1],  c="yellow", alpha=0.5)
	axarr[1][0].scatter(error_list_full[::, 4], error_list_full[::, 1],  c="yellow", alpha=0.5)
	axarr[1][0].scatter(error_list[ 4], error_list[ 1],  c="red", alpha=1.0)

	axarr[1][1].axis((start_time,end_time,57500,62500))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[1][1].scatter(train_list[::, 4], train_list[::, 2],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[1][1].scatter(train_list[::, 4], train_list[::, 2],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[1][1].scatter(train_list[::, 4], train_list[::, 2],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[1][1].scatter(train_list[::, 4], train_list[::, 2],  c="yellow", alpha=0.5)
	axarr[1][1].scatter(error_list_full[::, 4], error_list_full[::, 2],  c="yellow", alpha=0.5)
	axarr[1][1].scatter(error_list[4], error_list[2],  c="red", alpha=1.0)

	axarr[1][2].axis((start_time,end_time,0,1033))
	train_list = np.array(train_place_id[pred_list[0]])	
	axarr[1][2].scatter(train_list[::, 4], train_list[::, 3],  c="blue", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[2]])	
	axarr[1][2].scatter(train_list[::, 4], train_list[::, 3],  c="green", alpha=0.5)
	train_list = np.array(train_place_id[pred_list[4]])	
	axarr[1][2].scatter(train_list[::, 4], train_list[::, 3],  c="magenta", alpha=0.5)
	train_list = np.array(train_place_id[place_id])	
	axarr[1][2].scatter(train_list[::, 4], train_list[::, 3],  c="yellow", alpha=0.5)
	axarr[1][2].scatter(error_list_full[::, 4], error_list_full[::, 3],  c="yellow", alpha=0.5)
	axarr[1][2].scatter(error_list[4], error_list[3],  c="red", alpha=1.0)


	fig.savefig(plotfilename, format='pdf')

# Variance seems very high
# Preference to values nearer in time line
# Variance is low for place ids which happen throughout the time line while nearest timelines seems to have large variance in X
# Moving the second best to first would increase the percentage by 5% (since around 750 in 7000 odd samples)
# Moving the top 10 positions of 1024 to top 1 would increase percentage by 10 % (3-10 positions sums to around 700 ) 11-10 would increase by another 5 %
# The position 3 to 1 would increase by 2.5%
# Do not ignore all the train values with less number of place_ids since they may have corner cases where there are no other options and so may give better prediction than nothing
# Better logic is to use it in refining the options for chosing the top three. If there are multiple options and the top three has num

def get_max_record(error_dict):
	num_keys = len(error_dict.keys())
	keys = error_dict.keys()
	max_entry_info = np.empty((num_keys, 2), dtype="int")
	for idx in range(num_keys):
		max_entry_info[idx][0] = keys[idx]
		max_entry_info[idx][1] = len(error_dict[keys[idx]])
	max_entry_info.view('i8,i8').sort(order=['f1'], axis=0)
	return max_entry_info

def print_classification_metric(num_records_error):
	print "Classification metric"
	print num_records_error["zero"], num_records_error["one"], num_records_error["two"], num_records_error["none"], num_records_error["none_train"]
	total = num_records_error["zero"]+num_records_error["one"]+num_records_error["two"]+num_records_error["none"]+num_records_error["none_train"]
	mean = num_records_error["zero"]*1.0/total
	actual_metric = (num_records_error["zero"]*1.0 + num_records_error["one"]*0.5 + num_records_error["two"]*0.33)/total
	print mean, actual_metric

def calculate_distance(distances):
    distances[distances < .0001] = .0001
    return distances ** -2

def do_prediction(train_filename, eval_filename, pred_filename):
	classifier = RandomForestClassifier(n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1)
	#classifier = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)
	classifier = cf.get_trained_classifier(train_file, classifier)
	top_n_predicted = cf.get_prediction(classifier, eval_file)
	cf.dump_predicted_n(top_n_predicted, pred_file)
	pred_cost_position = cf.get_pred_actual_position(classifier, eval_file)
	return top_n_predicted, pred_cost_position
	# print pred_cost_position[:20]
	# print top_n_predicted[:20]


def plot_hist(filename, range_values, listvalue):
	fig = plt.figure()
	plt.axis(range_values)
	(n, bins, patches) = plt.hist(listvalue, bins=np.max(listvalue)+1)
	#print n[:10], bins[:10]
	fig.savefig(filename, format='pdf')
	return n, bins

def get_data_actual(filename, with_label):
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        feature = np.empty([num_entires, 4], dtype='i4')
        label = np.empty([num_entires, 1], dtype='i4').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [int(float(x)) for x in data_element]
            feature[row_id] = data_element[1:5]
            if(with_label):
                label[row_id] = data_element[5]
            row_id = row_id + 1
    if(with_label):
    	return (feature, label)
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

def get_prediction(classifier, feature):
    
    predict = classifier.predict_proba(feature)

    # argnmax is very slow compared to argmax. So better to replace value with negative value than Nan.
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
    return top_n_predicted

def get_train_set(classifier, feature, train_place_id):
	predict = classifier.predict_proba(feature)
	pred_pos = np.argsort(predict, axis=1)
	classes = classifier.classes_
	last_class = len(classes)-1
	top10_train = defaultdict(list);top20_train = defaultdict(list);top30_train = defaultdict(list)
	top10_train_list = [];top20_train_list = [];top30_train_list = [];
	for pred_order in classes[pred_pos]:
		for idx in range(30):
			place_id = pred_order[last_class - idx]
			if idx < 10:
				top10_train[place_id] = train_place_id[place_id]
			if idx < 20:
				top20_train[place_id] = train_place_id[place_id]
			if idx < 30:
				top30_train[place_id] = train_place_id[place_id]
	
	t10 = sum([len(value) for (key, value) in top10_train.iteritems()]);
	t20 = sum([len(value) for (key, value) in top20_train.iteritems()]);
	t30 = sum([len(value) for (key, value) in top30_train.iteritems()]);
	tot = sum([len(value) for (key, value) in train_place_id.iteritems()]);
	print len(top10_train.keys()), len(top20_train.keys()), len(top30_train.keys()), t10, t20, t30, tot
	return (top10_train, top20_train, top30_train)


	# axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="magenta", alpha=0.5)
	# axarr[0][0].scatter(train_list[::, 4], train_list[::, 1],  c="yellow", alpha=0.5)
	# axarr[0][0].scatter(error_list_full[::, 4], error_list_full[::, 1],  c="yellow", alpha=0.5)
	# axarr[0][0].scatter(error_value[3], error_value[0],  c="red", alpha=1.0)

def plot_error_top(eval_class, top_class, correct_class, plotfilename):
	eval_class = np.array(eval_class); top_class = np.array(top_class); correct_class = np.array(correct_class)
	fig = plt.figure()

	fig.subplots_adjust(left=0.05, right=0.99, bottom=0.05, top=0.99, hspace=0.1, wspace=0.1)
	axarr = []; num_graph = 0
	for i in range(2):
		axarr.append([])
		for j in range(3):
			axarr[i].append(fig.add_subplot(2,3,num_graph+1))
			num_graph = num_graph + 1

	#start_time = 786239*7/8; end_time = 786239
	hrs_in_week = 60*24*7
	hrs_in_day = 60*24
	# start_time = 0; end_time = hrs_in_day
	# top_class[::, 4] = (top_class[::, 4]+(60*18))%hrs_in_day
	# correct_class[::, 4] = (correct_class[::, 4]+(60*18))%hrs_in_day
	# eval_class[::, 4] = (eval_class[::, 4]+(60*18))%hrs_in_day

	hrs_in_month = 60*30*24
	offset = 0;#3*hrs_in_day/4;
	mod_value = hrs_in_day
	start_time = 0; end_time = mod_value
	top_class[::, 4] = (top_class[::, 4]+offset)%mod_value
	correct_class[::, 4] = (correct_class[::, 4]+offset)%mod_value
	eval_class[::, 4] = (eval_class[::, 4]+offset)%mod_value

	axarr[0][0].axis((start_time,end_time,0,100000))
	axarr[0][0].scatter(top_class[::, 4], top_class[::, 1],  c="blue", alpha=0.5)
	axarr[0][0].scatter(eval_class[::, 4], eval_class[::, 1],  c="red", alpha=0.5)

	axarr[0][1].axis((start_time,end_time,57500,62500))
	axarr[0][1].scatter(top_class[::, 4], top_class[::, 2],  c="blue", alpha=0.5)
	axarr[0][1].scatter(eval_class[::, 4], eval_class[::, 2],  c="red", alpha=0.5)

	axarr[0][2].axis((start_time,end_time,0,400))
	axarr[0][2].scatter(top_class[::, 4], top_class[::, 3],  c="blue", alpha=0.5)
	axarr[0][2].scatter(eval_class[::, 4], eval_class[::, 3],  c="red", alpha=0.5)

	axarr[1][0].axis((start_time,end_time,0,100000))
	axarr[1][0].scatter(correct_class[::, 4], correct_class[::, 1],  c="yellow", alpha=0.5)
	axarr[1][0].scatter(eval_class[::, 4], eval_class[::, 1],  c="red", alpha=0.5)

	axarr[1][1].axis((start_time,end_time,57500,62500))
	axarr[1][1].scatter(correct_class[::, 4], correct_class[::, 2],  c="yellow", alpha=0.5)
	axarr[1][1].scatter(eval_class[::, 4], eval_class[::, 2],  c="red", alpha=0.5)

	axarr[1][2].axis((start_time,end_time,0,400))
	axarr[1][2].scatter(correct_class[::, 4], correct_class[::, 3],  c="yellow", alpha=0.5)
	axarr[1][2].scatter(eval_class[::, 4], eval_class[::, 3],  c="red", alpha=0.5)

	fig.savefig(plotfilename, format='pdf')

def get_data_working(filename, is_train_data, offset_factor):
    NUM_FEATURES = 8
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        
        feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
        label = np.empty([num_entires, 1], dtype='f8').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [int(float(x)) for x in data_element]
            feature[row_id][0] = data_element[1]#/100000.0
            feature[row_id][1] = data_element[2]#/100000.0
            feature[row_id][2] = np.log2(data_element[3])*64 #/1033.0 #np.log()*100
            #feature[row_id][3] = (data_element[4]%(60*24))
            mod_value = (60*24.0); offset = offset_factor*mod_value
            time_offset = ((data_element[4]+offset) % mod_value)/mod_value
            feature[row_id][3] = (np.sin(2*np.pi*time_offset)+1)*mod_value/12 #10,11,13
            feature[row_id][4] = (np.cos(2*np.pi*time_offset)+1)*mod_value/12
            num_days = (data_element[4]//mod_value)
            weekday_offset = (num_days%7)/7
            feature[row_id][5] = (np.sin(2*np.pi*weekday_offset)+1)*13
            feature[row_id][6] = (np.cos(2*np.pi*weekday_offset)+1)*13
            	#(num_days%7)*27 #25,30 # There seems to be a influence by the number of weeks
            # yearday_offset = (num_days%365)/365
            # feature[row_id][7] = (np.sin(2*np.pi*yearday_offset)+1)*100
            # feature[row_id][8] = (np.cos(2*np.pi*yearday_offset)+1)*100
            feature[row_id][7] = (num_days//3) #2,4,5 # This feature gives importance to the placeid which are logged in recently
            # feature[row_id][7] = (num_days%30)*0.5 # There seems to be no influence from month
            # Since nearest neighbor seems to depend on distance, scale the inputs to meatch the dimension or importance of each feature
            # If adding a feature does not change the output metric then the feature is not given enough importance to have any significant impact
            # So try out values in the order of 1,10,100 till they start having impact. Once they start having impact, fine tune it by increasing or decreasing
            # to find out whether the output metric (mean) increases or decreases proportionaly or inversely. Then fine tune in that direction to get the apporpriate weights

            if(is_train_data):
                label[row_id] = data_element[5]
            row_id = row_id + 1
    if(is_train_data):
    	return (feature[:row_id], label[:row_id])
    else:
        return feature

def get_data(filename, is_train_data, offset_factor, scale_factor):
    NUM_FEATURES = 3
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        
        feature = np.empty([num_entires, NUM_FEATURES], dtype='f8')
        label = np.empty([num_entires, 1], dtype='f8').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [int(float(x)) for x in data_element]
            feature[row_id][0] = scale_factor[0]*(data_element[1]/10000.0)
            feature[row_id][1] = scale_factor[1]*(data_element[2]/10000.0)
            feature[row_id][2] = scale_factor[2]*np.log10(data_element[3])#/1033.0 #np.log()*100
            
            # mod_value = (60*24.0); offset = offset_factor*mod_value
            # time_offset = ((data_element[4]+offset) % mod_value)/mod_value
            # feature[row_id][3] = (np.sin(2*np.pi*time_offset)+1) #10,11,13
            # feature[row_id][4] = (np.cos(2*np.pi*time_offset)+1)
            # num_days = (data_element[4]//mod_value)
            # weekday_offset = (num_days%7)/7
            # feature[row_id][5] = (np.sin(2*np.pi*weekday_offset)+1)
            # feature[row_id][6] = (np.cos(2*np.pi*weekday_offset)+1)
            # 	#(num_days%7)*27 #25,30 # There seems to be a influence by the number of weeks
            # # yearday_offset = (num_days%365)/365
            # # feature[row_id][7] = (np.sin(2*np.pi*yearday_offset)+1)*100
            # # feature[row_id][8] = (np.cos(2*np.pi*yearday_offset)+1)*100
            # feature[row_id][7] = (num_days//3) #2,4,5 # This feature gives importance to the placeid which are logged in recently
            # # feature[row_id][7] = (num_days%30)*0.5 # There seems to be no influence from month
            # # Since nearest neighbor seems to depend on distance, scale the inputs to meatch the dimension or importance of each feature
            # # If adding a feature does not change the output metric then the feature is not given enough importance to have any significant impact
            # # So try out values in the order of 1,10,100 till they start having impact. Once they start having impact, fine tune it by increasing or decreasing
            # # to find out whether the output metric (mean) increases or decreases proportionaly or inversely. Then fine tune in that direction to get the apporpriate weights

            if(is_train_data):
                label[row_id] = data_element[5]
            row_id = row_id + 1
    if(is_train_data):
    	return (feature[:row_id], label[:row_id])
    else:
        return feature

# Final Max Mean and sf 0.396519754671 [1.0, 2, 0.0375] Max Score and sf 0.492376265868 [1.0, 2.5, 0.05]
def finding_optimal_parameters(train_file, eval_file):
	max_mean = 0.0; max_score = 0.0; max_mean_sf = []; max_score_sf = []; nn_val_mean = 0; nn_val_score = 0;
	# for scale_y in [ 2, 2.25, 2.5, 2.75, 3]:
	# 	for scale_a in [ 0.025, 0.0375, 0.05, 0.0625, 0.075]:
	for n_neighbor in [50, 75, 100, 200, 250, 275, 500]:
			#sf = [1.0, scale_y, scale_a]
			sf = [1.0, 2.5, 0.05]
			trf, trl = get_data(train_file, True, 0, sf)
			evf, evl = get_data(eval_file, True, 0, sf)
			knc = KNeighborsClassifier(n_neighbors=n_neighbor, weights=calculate_distance, metric='manhattan', n_jobs = -1)
			knc.fit(trf, trl)
			knc_pcp = get_pred_actual_position(knc, evf, evl)
			(knc_n, bins, patches) = plt.hist(knc_pcp[:,2], bins=np.max(knc_pcp[:,2])+1)
			mean = knc_n[1]/np.sum(knc_n); score = (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])/np.sum(knc_n)
			print "sf", sf, "top1,2,3", knc_n[1], knc_n[2], np.sum(knc_n[1:4]), "top-1,10,20,30", knc_n[0], np.sum(knc_n[1:10]), np.sum(knc_n[1:20]), np.sum(knc_n[1:30])
			print "mean", mean, "score", score
			if(mean > max_mean):
				max_mean = mean; max_mean_sf = sf; nn_val_mean = n_neighbor
			if(score > max_score):
				max_score = score; max_score_sf = sf; nn_val_score = n_neighbor

	print "Final", "Max Mean and sf", max_mean, max_mean_sf, nn_val_mean, "Max Score and sf", max_score, max_score_sf, nn_val_score

def get_place_id_grouping(feature_list, label_list):
	unique_id = defaultdict(list);
	idx = 0
	for feature in feature_list:
		unique_id[label_list[idx]].append(feature)
		idx += 1
	return unique_id

def multi_classifier(train_file, eval_file):
	NUM_CLASSIFIER = 4
	t3p = NUM_CLASSIFIER*[0]
	for idx in range(NUM_CLASSIFIER):
		trf, trl = get_data(train_file, True, idx*.25)
		evf, evl = get_data(eval_file, True, idx*.25)
		knc = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)
		knc.fit(trf, trl)
		t3p[idx] = get_prediction(knc, evf)
		compare = t3p[idx][:,0]==evl
		print compare.mean();
	class_value = np.vstack((t3p[0][:,0], t3p[1][:,0], t3p[2][:,0], t3p[3][:,0])).transpose()
	cost_value = np.vstack((t3p[0][:,1], t3p[1][:,1], t3p[2][:,1], t3p[3][:,1])).transpose()
	class_idx = np.argmax(cost_value, axis=1)
	print cost_value.shape, class_idx.shape, class_idx[:10]
	min_class = class_value[range(7011), class_idx]
	print min_class.shape, min_class
	evf, evl = get_data(eval_file, True, 0)
	print (min_class==evl).mean()


# 1023 79.6675789356 (7011,)
# 0.443303380402

def kde_classifier(train_file, eval_file):
	trf, trl = get_data(train_file, True, 0)
	evf, evl = get_data(eval_file, True, 0)
	tpi = get_place_id_grouping(trf, trl)
	start_time = time.time()
	kde_kernals = {}
	pred_prob = np.zeros(( len(tpi.keys()),  len(evf) ))
	classes = np.zeros(len(tpi.keys()))
	idx = 0
	for place_id, train_list in tpi.iteritems():
		# print train_list, evf
		kde_kernals[place_id] = KernelDensity(kernel='gaussian', bandwidth=100).fit(train_list)
		pred_prob[idx] = kde_kernals[place_id].score_samples(evf)
		classes[idx] = place_id
		# print pred_prob[idx]
		idx += 1
	min_array = np.argmax(pred_prob, axis=0)
	end_time = time.time() - start_time
	print len(tpi.keys()), end_time, min_array.shape
	pred_label = classes[min_array]
	print (pred_label==evl).mean()
	print pred_label[:10], evl[:10], pred_prob[:10, :10]


def kde_single_classifier(train_file, eval_file):
	trf, trl = get_data(train_file, True, 0)
	evf, evl = get_data(eval_file, True, 0)
	tpi = get_place_id_grouping(trf, trl)
	start_time = time.time()
	pred_prob = np.zeros(( len(tpi.keys()),  len(evf) ))
	classes = np.zeros(len(tpi.keys()))
	oidx = 0
	for place_id, train_list in tpi.iteritems():
		# print train_list, evf
		train_list = np.array(train_list)
		# print train_list[::,0].shape, evf[::,0].shape
		pred_prob_cost = np.zeros(len(evf)) 
		for idx in range(4):
			train = train_list[::,idx].reshape(-1,1)
			eval_val = evf[::,idx].reshape(-1,1)
			# print train, eval_val
			classifier = KernelDensity(kernel='gaussian', bandwidth=100).fit(train)
			cost = classifier.score_samples(eval_val)
			#print cost
			pred_prob_cost = np.add(pred_prob_cost, cost)
			#print pred_prob_cost
		pred_prob[oidx] = pred_prob_cost
		classes[oidx] = place_id
		if oidx%100 == 0:
			print "pred_prob", oidx, classes[oidx], pred_prob[oidx]
		# print pred_prob[idx]
		oidx += 1
	min_array = np.argmax(pred_prob, axis=0)
	end_time = time.time() - start_time
	print len(tpi.keys()), end_time, min_array.shape
	pred_label = classes[min_array]
	print (pred_label==evl).mean()
	print pred_label[:10], evl[:10], min_array[:10], pred_prob[:10, :10]


def kde_refinement_classifier(train_file, eval_file):
	trf, trl = get_data(train_file, True, 0)
	evf, evl = get_data(eval_file, True, 0)

	# First level classification	
	knc = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)
	knc.fit(trf, trl)
	predict = knc.predict_proba(evf)
	pred_pos = np.argsort(predict, axis=1)
	classes = classifier.classes_
	pred_classes = classes#Copy classes for all the selection

	# Top N prediction from kNN
	NUM_PREDICTION = 11
	num_samples = len(evf)
	for sample_idx in range(num_samples):
		for pred_opt_idx in range(NUM_PREDICTION):
			print "hi"

	tpi = get_place_id_grouping(trf, trl)
	start_time = time.time()
	pred_prob = np.zeros(( len(tpi.keys()),  len(evf) ))
	classes = np.zeros(len(tpi.keys()))
	oidx = 0
	for place_id, train_list in tpi.iteritems():
		# print train_list, evf
		train_list = np.array(train_list)
		# print train_list[::,0].shape, evf[::,0].shape
		pred_prob_cost = np.zeros(len(evf)) 
		for idx in range(4):
			train = train_list[::,idx].reshape(-1,1)
			eval_val = evf[::,idx].reshape(-1,1)
			# print train, eval_val
			classifier = KernelDensity(kernel='gaussian', bandwidth=100).fit(train)
			cost = classifier.score_samples(eval_val)
			#print cost
			pred_prob_cost = np.add(pred_prob_cost, cost)
			#print pred_prob_cost
		pred_prob[oidx] = pred_prob_cost
		classes[oidx] = place_id
		if oidx%100 == 0:
			print "pred_prob", oidx, classes[oidx], pred_prob[oidx]
		# print pred_prob[idx]
		oidx += 1
	min_array = np.argmax(pred_prob, axis=0)
	end_time = time.time() - start_time
	print len(tpi.keys()), end_time, min_array.shape
	pred_label = classes[min_array]
	print (pred_label==evl).mean()
	print pred_label[:10], evl[:10], min_array[:10], pred_prob[:10, :10]






# Total Eval Samples 7011, total classes 1023
# RFC (RandomForest) / KNC (KNearestNeighbor)
# (7 Features) top 3241.0 2878.0 2nd 751.0 1011.0 top3 4348.0 4328.0 no idx 539.0 539.0 top10 4971.0 5189.0 top20 5346.0 5513.0
# (3 Features) top 2646.0 2780.0 2nd 945.0 1002.0 top3 4055.0 4277.0 no idx 539.0 539.0 top10 4979.0 5178.0 top20 5300.0 5503.0 top30 5342.0 5577.0
# Except top in all positions KNN performs better than Random Forest
# KNN performs almost same with just three features instead of 7 features
# top3 gives 61%, top10 73.85% Accuracy, top20  78.49% Accuracy, top30 seems to achieve 79.54%

# KNN
# (n_neighbors=100)  top1,2,3 2801.0 971.0  4263.0 top-1,10,20,30 539.0 5182.0 5516.0 5582.0
# (n_neighbors=200)  top1,2,3 2790.0 1000.0 4280.0 top-1,10,20,30 539.0 5214.0 5574.0 5716.0
# (n_neighbors=1000) top1,2,3 2701.0 1002.0 4219.0 top-1,10,20,30 539.0 5133.0 5616.0 5796.0

# with 4 features ((data_element[4]%(60*24))*10)
# (n_neighbors=1000) top1,2,3 2629.0 790.0 3831.0 top-1,10,20,30 539.0 4798.0 5269.0 5518.0
# (data_element[4]%(60*24))
# (n_neighbors=1000) top1,2,3 2934.0 878.0 4239.0 top-1,10,20,30 539.0 5174.0 5593.0 5763.0
# (n_neighbors=500) top1,2,3 3050.0 889.0 4390.0 top-1,10,20,30 539.0 5263.0 5625.0 5766.0
# (n_neighbors=100) top1,2,3 3202.0 931.0 4536.0 top-1,10,20,30 539.0 5287.0 5525.0 5601.0
# without log scaling of accuracy
# (n_neighbors=100) top1,2,3 3210.0 887.0 4511.0 top-1,10,20,30 539.0 5263.0 5533.0 5618.0
# without weights=calculate_distance and metric='manhattan'
# (n_neighbors=100) top1,2,3 3114.0 894.0 4444.0 top-1,10,20,30 539.0 5248.0 5509.0 5589.0

# with time offset 18*
# top1,2,3 3229.0 870.0 4514.0 top-1,10,20,30 539.0 5272.0 5531.0 5625.0





# (n_neighbors=1000) top1,2,3 2701.0 1002.0 4219.0 top-1,10,20,30 539.0 5133.0 5616.0 5796.0


if __name__ == "__main__":
	XY_SD_SP_EVAL_TS_2SD_PRED_PATH = "../data/xy_sd_split/eval_ts_2sd_10000x_1500y/pred_try1/xys_ev_pr" #5_39
	eval_file = ct.XY_SD_SP_EVAL_TS_2SD_PATH+"1_40.csv"
	train_file = ct.XY_SD_SP_TR_TS_2SD_PATH+"1_40.csv"

	pred_file = XY_SD_SP_EVAL_TS_2SD_PRED_PATH+"5_39_temp.csv"

	# multi_classifier(train_file, eval_file)
	# kde_classifier(train_file, eval_file)
	# kde_single_classifier(train_file, eval_file)


	finding_optimal_parameters(train_file, eval_file)


	# trf, trl = get_data(train_file, True, 0)
	# evf, evl = get_data(eval_file, True, 0)
	# trfa, trla = get_data_actual(train_file, True)
	# evfa, evla = get_data_actual(eval_file, True)

	# tpi, tpin = get_unique_place_id(train_file, 1, None)
	# epi, epin = get_unique_place_id(eval_file, 1, None)

	# # rfc = RandomForestClassifier(n_estimators=20, max_features=3, class_weight="balanced", max_depth=20, n_jobs = -1)
	# # rfc.fit(trf, trl)
	# # rfc_pred_cost_position = get_pred_actual_position(rfc, eval_file)
	# # (rfc_n, bins, patches) = plt.hist(rfc_pred_cost_position[:,2], bins=np.max(rfc_pred_cost_position[:,2])+1)

	# #knc = KNeighborsClassifier(n_neighbors=100, n_jobs = -1)
	# knc = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)
	# # knc = GMM(n_components=1023, covariance_type='tied', init_params='wc', n_iter=20)
	# #knc = GaussianNB()
	# # knc = RandomForestClassifier(n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1)
	# knc.fit(trf, trl)
	# knc_pcp = get_pred_actual_position(knc, evf, evl)
	# (knc_n, bins, patches) = plt.hist(knc_pcp[:,2], bins=np.max(knc_pcp[:,2])+1)

	# print "top1,2,3", knc_n[1], knc_n[2], np.sum(knc_n[1:4]), "top-1,10,20,30", knc_n[0], np.sum(knc_n[1:10]), np.sum(knc_n[1:20]), np.sum(knc_n[1:30])
	# print "mean", knc_n[1]/np.sum(knc_n), "score", (knc_n[1]+0.5*knc_n[2]+0.33*knc_n[3])/np.sum(knc_n)
	# print "top", rfc_n[1], knc_n[1], "2nd", rfc_n[2], knc_n[2], "top3", np.sum(rfc_n[1:4]), np.sum(knc_n[1:4]), 
	# print "no idx", rfc_n[0], knc_n[0], "top10", np.sum(rfc_n[1:10]),  np.sum(knc_n[1:10]), "top20", np.sum(rfc_n[1:20]),  np.sum(knc_n[1:20]),
	# print "top30", np.sum(rfc_n[1:30]),  np.sum(knc_n[1:30])

	# top_n_predicted = get_prediction(knc, evf)

	# (zero_option, first_option, second_option, not_in_list, not_in_train, not_in_list_pred, num_records_error) = get_error_records_list(eval_file, top_n_predicted, tpi, knc_pcp)

	# # (top10_train, top20_train, top30_train) = get_train_set(knc, evf, tpi)
	# num_val_small = 0; num_val_big = 0
	# num_graphs = 0
	# second_class = []
	# for i in range(len(knc_pcp)):
	# 	if(knc_pcp[i][2] == 1):
	# 		# if(knc_pcp[i][1] == 183.0 or knc_pcp[i][1] == 808.0 or knc_pcp[i][1] == 110.0):
	# 		if(knc_pcp[i][1] == 478.0 or knc_pcp[i][1] == 337.0 or knc_pcp[i][1] == 132.0 or knc_pcp[i][1] == 110.0):
	# 			print knc_pcp[i][0], knc_pcp[i][1], top_n_predicted[i][0], top_n_predicted[i][1], top_n_predicted[i][3], 
	# 			print evfa[i], len(tpi[top_n_predicted[i][2]]), len(tpi[top_n_predicted[i][0]])
	# 		second_class.append(knc_pcp[i][1])
			
	# 		# if num_graphs == 20:
	# 		# 	plotfilename = "../data/plots/"+str(num_graphs)+"_"+str(evla[i])+".pdf"
	# 		# 	plot_error_top(evfa[i], tpi[top_n_predicted[i][2]], tpi[top_n_predicted[i][0]], plotfilename)
	# 		# 	num_graphs += 1
	# 			#break
	# 		# if( len(tpi[top_n_predicted[i][2]]) < len(tpi[top_n_predicted[i][0]])):
	# 		# 	num_val_small += 1
	# 		# else:
	# 		# 	num_val_big += 1
	# print num_val_small, num_val_big, len(second_class)
	# n,bins = plot_hist("../data/plots/pred_pos_rest_hist.pdf", (0,1023,0,75), second_class)

	# hist_array = np.array(n)
	# sorted_out = np.argsort(hist_array)[-10:]
	# print hist_array[sorted_out], bins[sorted_out]

	# plot_error_top(epi[21733], tpi[21733], tpi[22251], "../data/plots/183_21733_22251.pdf")
	# plot_error_top(epi[88200], tpi[88200], tpi[31729], "../data/plots/808_88200_31729.pdf")
	# plot_error_top(epi[24040], tpi[24040], tpi[10440], "../data/plots/209_24040_10440.pdf")
	# plot_error_top(epi[24040], tpi[24040], tpi[13941], "../data/plots/209_24040_13941.pdf")

	# plot_error_top(epi[53669], tpi[53669], tpi[94805], "../data/plots/478_53669_13941.pdf")
	# plot_error_top(epi[37987], tpi[37987], tpi[4163] , "../data/plots/337_37987_4163.pdf")
	# plot_error_top(epi[15810], tpi[15810], tpi[32790] , "../data/plots/132_15810_32790.pdf")



	# print "do prediction"
	# top_n_predicted, pred_cost_position = do_prediction(train_file, eval_file, pred_file, train_place_id_num)

	# plot_hist("../data/plots/pred_pos_rest_hist.pdf", (3,30,0,200), pred_cost_position[:,2])
	# plot_hist("../data/plots/pred_pos_top3_hist.pdf", (0,10,0,4000), pred_cost_position[:,2])
	# #plt.clf()

	# (missing_place_id, present_place_id, num_missing_record, num_present_record) =  get_missing_place_id(eval_place_id, train_place_id)
	# missing_record_ratio = num_missing_record*1.0/(num_missing_record+num_present_record)
	# print "missing record", len(missing_place_id), len(eval_place_id.keys()), num_missing_record, num_present_record, missing_record_ratio

	# (zero_option, first_option, second_option, not_in_list, not_in_train, not_in_list_pred, num_records_error) = get_error_records_list(eval_file, top_n_predicted, train_place_id, pred_cost_position)
	# print len(zero_option.keys()), len(first_option.keys()), len(second_option.keys()), len(not_in_list.keys())
	# print_classification_metric(num_records_error)

	# NUM_GRAPHS = 20
	# max_entry_info = get_max_record(not_in_list)
	# for i in range(1,NUM_GRAPHS+1):
	# 	num_error = max_entry_info[-1*i][1]
	# 	error_place_id = max_entry_info[-1*i][0]
	# 	num_train_samples = len(train_place_id[error_place_id])
	# 	plot_file = "../data/plots/"+str(i)+"_"+str(num_error)+"_"+str(num_train_samples)+"_"+str(error_place_id)+".pdf"
	# 	plot_options(train_place_id, not_in_list, not_in_list_pred, error_place_id, plot_file)


	# 
	# print "Graph Details"
	# NUM_GRAPHS = 20
	# 
	# 	place_id = max_entry_info[-1*i][0]#not_in_list_keys[idx]
	# 	
	# 	num_train_samples = len(train_place_id[place_id])
	# 	print len(not_in_list[place_id]), len(train_place_id[place_id]), len(zero_option[place_id]), len(first_option[place_id]), len(second_option[place_id]), place_id
	# 	train_list = np.array(train_place_id[place_id])
	# 	eval_list = np.array(not_in_list[place_id])
	# 	plot_file = "../data/plots/"+str(num_error)+"_"+str(num_train_samples)+"_"+str(place_id)+".pdf"
	# 	plot_graph(train_list, eval_list, plot_file);
	#print train_list
