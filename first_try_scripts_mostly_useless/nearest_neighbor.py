
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import time
import heapq



#FILE_PATH = "../data/fb/train/xaa1_modified"



#MEAN_STATISTICS = "/Volumes/movies/test_data/fb/mean_statistics.csv"
MEAN_STATISTICS = "../data/fb/train/mean_statistics.csv"
#IDX_TO_ID_FILENAME = "/Volumes/movies/test_data/fb/id_to_idx"
IDX_TO_ID_FILENAME = "../data/fb/train/id_to_idx"
TEST_FILE = "/Volumes/movies/test_data/fb/test.csv"
#TEST_FILE = "../data/fb/train/xaa1"
RESULT_FILE = "/Volumes/movies/test_data/fb/test_output.csv"
#RESULT_FILE = "../data/fb/train/xaa1_result"


def get_stats(filename):
	with open(filename) as infile:
		stats = []
		for line in islice(infile,1,None):
			data_element = line.strip().split(",")
			data_element = [float(x) for x in data_element]
			stats.append(data_element)
	np_stats = np.array(stats)
	return np_stats

def get_idx_to_id(filename):
	with open(filename) as infile:
		idx_to_id = []
		for line in infile:
			idx_to_id.append(line.strip())
	np_idx_to_id = np.array(idx_to_id)
	return np_idx_to_id


def get_test(instring):
	np_input = np.array([float(x) for x in instring.strip().split(",")])
	np_input_norm = np_input[1:5]/[10.0,10.0,1033.0,786239.0]
	return (np_input_norm)

def get_cost(test, eval_value):
	# np.sqrt(np.sum((get_test(line) - mean_stat[0][:4]/mean_stat[0][8])**2))
	return np.linalg.norm(test[:4] - (eval_value[:4]/eval_value[8]))
	#return np.linalg.norm(test[:2] - (eval_value[:2]/eval_value[8]))

def get_cost_array(test, mean_stat):
	difference = mean_stat - test
	#print difference
	return np.linalg.norm(difference, axis=1)
	#return np.linalg.norm(test[:2] - (eval_value[:2]/eval_value[8]))

def get_min_idx(cost_array):
	min0 = 9999999;min1 = 9999999;min2 = 9999999;
	min0_idx = -1;min1_idx = -1;min2_idx = -1;
	for idx in range(cost_array.shape[0]):
		cost = cost_array[idx] #get_cost(test, mean_stat[idx])
		#if(cost_array[idx] == get_cost(test, mean_stat[idx])):
		#	print "Difference",cost_array[idx],get_cost(test, mean_stat[idx])
		if(cost < min0):
			min2 = min1; min2_idx = min1_idx;
			min1 = min0; min1_idx = min0_idx;
			min0 = cost; min0_idx = idx;
			continue
		if(cost < min1):
			min2 = min1; min2_idx = min1_idx;
			min1 = cost; min1_idx = idx;
			continue
		if(cost < min2):
			min2 = cost; min2_idx = idx;
			continue
	#return (min0, min0_idx, min1, min1_idx, min2, min2_idx)
	return (min0_idx, min1_idx, min2_idx)

def get_min_cost(test, mean_stat):
	#print test.shape, mean_stat.shape
	cost_array = get_cost_array(test, mean_stat)
	#print get_min_idx(cost_array), cost_array.argsort()[:3]
	#indices = heapq.nsmallest(3,np.nditer(cost_array),key=cost_array.__getitem__)
	return cost_array.argsort()[:3]

def get_output_string(row_id, idx_list, idx_to_id):
	return str(row_id)+","+str(idx_to_id[idx_list[0]])+" "+str(idx_to_id[idx_list[1]])+" "+str(idx_to_id[idx_list[2]])+"\n"

mean_stat = get_stats(MEAN_STATISTICS)
idx_to_id = get_idx_to_id(IDX_TO_ID_FILENAME)
print mean_stat.shape

mean_stat = np.transpose(mean_stat)
mean_stat = np.transpose(mean_stat/mean_stat[8])[:,:4]
#print mean_stat[:,8]

with open(TEST_FILE) as infile:
	with open(RESULT_FILE,"w") as outfile:
		outfile.write("row_id,place_id\n")
		row_id = 0
		start_time = time.time()
	 	for line in islice(infile,1,None):
	 		output = get_min_cost(get_test(line), mean_stat)
	 		#print output, idx_to_id[output[0]]	
	 		outfile.write( get_output_string(row_id, output, idx_to_id))
	 		row_id = row_id + 1
	 		if(row_id%1000 == False):
	 			duration = time.time() - start_time
	 			print row_id, duration
	 			start_time = time.time()



































