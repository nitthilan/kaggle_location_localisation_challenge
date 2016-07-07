#!/usr/bin/python

from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import gzip
import shutil
import os
import math


from os import listdir
from os.path import isfile, join

import constants as ct
import utils as ut

#https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms

# Rough size of input data 1.27GB time taken for a 10Mb file 33 Sec
# Therefor total time for one iteration 123.91898828125 * 33 = 4089.32661328125 (4096) seconds
# One iteration would take around 69 minutes

# The way the evaluation is done:
# Each test observation has exactly 1 correct prediction: the place where the checkin effectively 
# took place when the data generating simulation was run. You are allowed up to 3 predictions. 
# If your first prediction is right, you get 1/1, if your 2nd prediction is right, you get 1/2, 
# and if your 3rd prediction is right, your get 1/3. Otherwise, you get 0 for the particular observation. 
# Hence, there is no reason not to make 3 predictions per test observation, in the order of likelihood.

# Time is shown in minutes
# https://www.kaggle.com/chomolungma/facebook-v-predicting-check-ins/time-is-in-minutes-simple-reseaoning/discussion

# Very interesting analysis of the data:
# https://www.kaggle.com/msjgriffiths/facebook-v-predicting-check-ins/exploratory-data-analysis/comments

def min_max_total(xmin, xmax, xtotal, xsqrtotal, x):
	x = float(x)
	if(x < xmin): xmin = x
	if(x > xmax): xmax = x
	xtotal = xtotal + x
	xsqrtotal = xsqrtotal + x*x
	return (xmin, xmax, xtotal, xsqrtotal)

# print num_entries, len(unique_entries.keys()), min(unique_entries.values()), max(unique_entries.values()), sum(unique_entries.values())
# 29118022 108390 1 1849 29118021
# min, max, sum, sqrsum, mean, variance (std_deviation)
# x 0.0 100000.0 1.45583218023e+12 9.65655984118e+16 49997.633089 816588270.502 (28875.046)
# y 0.0 100000.0 1.45642737658e+12 9.71253765874e+16 50018.0739124 833768324.566 (28875.0467)
# a 1.0 1033.0 2412402560.0 5.83290633356e+11 82.8491221004 13167.9685903 (114.751769)
# t 1.0 786239.0 1.21425165572e+13 6.61969244232e+18 417010.350402 53442413019.3 (231176.151)

# Test
# 8607231
# x 0.0 100000.0 4.29622153113e+11 2.85161592468e+16 49914.0958472 821629850.32
# y 0.0 100000.0 4.30938061925e+11 2.87491404793e+16 50066.9799527 833412414.759
# a 1.0 1026.0 797477730.0 2.06853788674e+11 92.6520654552 15448.1539806
# t 786242.0 1006589.0 7.66442554218e+12 6.86066494628e+18 890463.558161 4156192244.82


def get_basic_stat(filename, isTrain):
	#Reading large files
	#http://stackoverflow.com/questions/6475328/read-large-text-files-in-python-line-by-line-without-loading-it-in-to-memory
	with open(filename) as infile:
		unique_entries = {}; num_entries = 0; unique_entries = {}
		# 0/1/2/3 - min, max, sum, sqrsum
		# 0/1/2/3 - x, y, accuracy, time
		full_data_stat = []
		
		# Initialise array to zero
		for i in range(4):
			full_data_stat.append([]);
			for j in range(4):
				full_data_stat[i].append(0.0)
		
		# Initialise min value to a higher value
		for i in range(4):
			full_data_stat[i][0] = 1000000.0;

		for line in islice(infile,1,None):
			num_entries = num_entries + 1
			data_element = line.strip().split(",")
			#print data_element
			if(isTrain):
				place_id = data_element[5]
				if(unique_entries.has_key(place_id)):
					unique_entries[place_id] = unique_entries[place_id] + 1;
				else:
					unique_entries[place_id] = 1;
			for i in range(4):
				(full_data_stat[i][0], full_data_stat[i][1], full_data_stat[i][2], full_data_stat[i][3]) = \
					min_max_total(full_data_stat[i][0], full_data_stat[i][1], full_data_stat[i][2], full_data_stat[i][3], data_element[i+1])
		num_entries = num_entries + 1
			
		# Calculating mean and std deviation
		mean = []; std_deviation = []
		for i in range(4):
			mean.append(full_data_stat[i][2]/num_entries)
			std_deviation.append(full_data_stat[i][3]/num_entries - (mean[i]**2))

		if isTrain:
			print num_entries, len(unique_entries.keys()), min(unique_entries.values()), max(unique_entries.values()), sum(unique_entries.values())
		else:
			print num_entries
		tag = ["x","y","a","t"]
		for i in range(4):
			print tag[i], full_data_stat[i][0], full_data_stat[i][1], full_data_stat[i][2], full_data_stat[i][3], mean[i], std_deviation[i]

		if isTrain:
			return (num_entries, full_data_stat, unique_entries)
		else:
			return (num_entries, full_data_stat)
# Data for x an y has only 4 decimal precision
# so Range is from 0 - 100000
# Accuracy range is 1033
# Time series - 786239
def convert_data_to_int(infilename, outfilename, is_train):
	with open(infilename) as infile:
		with open(outfilename, "w") as outfile:
			if(is_train):
				outfile.write("row_id,x,y,accuracy,time,place_id\n")
			else:
				outfile.write("row_id,x,y,accuracy,time\n")
			unique_id = 0; is_unique = {}; unique_array = []
			for line in islice(infile,1,None):
				data_element = [float(x) for x in line.strip().split(",")]
				record_id = int(data_element[0])
				x = int(data_element[1]*10000)
				y = int(data_element[2]*10000)
				accuracy = int(data_element[3])
				place_id = 0
				if(is_train):
					place_id = int(data_element[5])
					if(is_unique.has_key(place_id)):
						uid = is_unique[place_id]
						unique_array[uid][1] = unique_array[uid][1] + 1;
					else:
						is_unique[place_id] = unique_id
						unique_array.append([place_id, 1, 0, 0])
						unique_id = unique_id + 1
				outstring = str(record_id)+","+str(x)+","+str(y)+","+str(accuracy)+","+str(int(data_element[4]))
				if(is_train):
					outstring = outstring + ","+str(is_unique[place_id])
				outstring = outstring + "\n"
				outfile.write(outstring)
	return unique_array


MAX_TIME_STAMP = 786242.0
# eval 6906652  train 22211369 for 0.8
# eval 10273021 train 18845000 for 0.7
# eval 3544819  train 25573202 for 0.9
# eval 5254272  train 23863749 for 0.85 (roughly 22%)
def split_train_record(trainfilename, splittrainfilename, splitevalfilename):
	num_train_records = 0; num_eval_records = 0;
	with open(trainfilename) as trainfile, open(splittrainfilename, "w") as splitfile, open(splitevalfilename, "w") as evalfile:
		splitfile.write("row_id,x,y,accuracy,time,place_id\n");
		evalfile.write("row_id,x,y,accuracy,time,place_id\n");
		for line in islice(trainfile, 1, None):
			data_record = line.strip().split(",")
			timestamp = float(data_record[4])
			if(timestamp > 0.85 * MAX_TIME_STAMP):
				evalfile.write(line)
				num_eval_records = num_eval_records + 1
			else:
				splitfile.write(line)
				num_train_records = num_train_records + 1
	print num_eval_records, num_train_records


# Num unique entries for SPlit Train 106435
def dump_stats(infilename, outfilename):
	num_features = 7
	with open(infilename) as infile:
		#stat = np.zeros((num_unique, num_features), dtype=np.int)
		stat = {}
		for line in islice(infile, 1, None):
			data_record = [int(float(x)) for x in line.strip().split(",")]
			place_id = data_record[5]
			if(not stat.has_key(place_id)):
				stat[place_id] = [0,0,0,0,0,0,0]

			stat[place_id][0] = stat[place_id][0] + data_record[1]
			stat[place_id][1] = stat[place_id][1] + data_record[2]
			stat[place_id][2] = stat[place_id][2] + data_record[3]
			stat[place_id][3] = stat[place_id][3] + data_record[1]*data_record[1]
			stat[place_id][4] = stat[place_id][4] + data_record[2]*data_record[2]
			stat[place_id][5] = stat[place_id][5] + data_record[3]*data_record[3]
			stat[place_id][6] = stat[place_id][6] + 1
	print len(stat.keys())

	with open(outfilename, "w") as outfile:
		outfile.write("id,sum_x,sum_y,sum_accuracy\n")
		for idx in stat.keys():
			outstring = str(idx)
			for fid in range(num_features):
				outstring = outstring + ","+str(stat[idx][fid])
			outstring = outstring + "\n"
			outfile.write(outstring)
			
def store_unique(unique_array, outfilename):
	with open(outfilename, "w") as outfile:
		outfile.write("label,num_entries\n")
		for idx in range(len(unique_array)):
			outfile.write(str(unique_array[idx][0])+","+str(unique_array[idx][1])+"\n")

# Average 1-sd (std deviation) implies 70% of values would lie between +/- 1sd
# 108391 6750.0715741 206.229965204 106.961287001
# Average 2-sd imples 95% values lie between +/- 2sd
# 108391 13500.1431482 412.459930408 213.922574002
# Average 3-sd (std deviation) implies 99% of values would lie between +/- 3sd
# 108391 20250.2147223 618.689895612 320.883861004

def analyse_stats(uniqstatfilename):
	with open(uniqstatfilename) as infile:
		avg_std_dev = [0.0, 0.0, 0.0]
		total_unique_entries = 1
		for line in islice(infile,1,None):
			data_record = [(float(x)) for x in line.strip().split(",")]
			num_entries = data_record[7]
			for idx in range(3):
				mean = data_record[idx+1]/num_entries
				sqrmean = data_record[idx+4]/num_entries
				std_dev = math.sqrt(sqrmean - mean**2)
				#print idx, data_record[idx+1], mean, data_record[idx+4], sqrmean, std_dev, num_entries
				avg_std_dev[idx] = avg_std_dev[idx] + 2*std_dev
			total_unique_entries = total_unique_entries + 1
		print total_unique_entries, avg_std_dev[0]/total_unique_entries, avg_std_dev[1]/total_unique_entries, avg_std_dev[2]/total_unique_entries


#unique_array = convert_data_to_int(ct.TRAIN_FILE_PATH, ct.TRAIN_ALL_INT_PATH, True)
#get_basic_stat(ct.TRAIN_ALL_INT_PATH)
#store_unique(unique_array, ct.UNIQUE_ID_PATH)
#convert_data_to_int(ct.TEST_FILE_PATH, ct.TEST_ALL_INT_PATH, False)
#print "start"
#dump_stats(ct.TRAIN_ALL_INT_PATH, 108390, ct.STATS_PATH)
#analyse_stats(ct.STATS_PATH)

#get_basic_stat(ct.TEST_ALL_INT_PATH, False)

#split_train_record(ct.TRAIN_ALL_INT_PATH, ct.SPLIT_TRAIN_ALL_INT_PATH, ct.SPLIT_EVAL_ALL_INT_PATH)
dump_stats(ct.SPLIT_TRAIN_ALL_INT_PATH, ct.SPLIT_STATS_PATH)

