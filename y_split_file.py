#!/usr/bin/python

from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import gzip
import shutil
import os

from os import listdir
from os.path import isfile, join

import constants as ct
import utils as ut

def get_num_range():
	file_list = range(ct.Y_STEP_SIZE,ct.MAX_Y_VALUE-2*ct.Y_STEP_SIZE, ct.Y_STEP_SIZE)
	start_list = range(0,ct.MAX_Y_VALUE-ct.Y_RANGE, ct.Y_STEP_SIZE)
	end_list = range(ct.Y_RANGE-1,ct.MAX_Y_VALUE, ct.Y_STEP_SIZE)
	end_list[-1] = ct.MAX_Y_VALUE
	# print len(file_list), len(start_list), len(end_list), file_list[-1], start_list[-1], end_list[-1]
	# list1 = np.array(range(100))
	# print file_list, start_list, end_list
	# print 3*(list1/45)
	# print 3*((list1-15)/45)+1
	# print 3*((list1-30)/45)+2
	# print list1/15
	return len(start_list) #(start_list, end_list)

def get_fileid_array(filename_base, num_files, is_train):
	fileid_array = []
	for idx in range(num_files):
		filename = filename_base+str(idx)+".csv"
		with open(filename, "w") as infile:
			if(is_train):
				infile.write("row_id,x,y,accuracy,time,place_id\n");
			else:
				infile.write("row_id,x,y,accuracy,time\n");
		fileid = open(filename, "a")
		fileid_array.append(fileid)
	return fileid_array

def close_fileid_array(fileid_array):
	for fileid in fileid_array:
		fileid.close()

def write_file(fileid_array, num_records_per_file,line, yfilter, num_files):
	if(yfilter >= 0 and yfilter < num_files):
		fileid_array[yfilter].write(line)
		num_records_per_file[yfilter] = num_records_per_file[yfilter] + 1
	return

def split_file_train(input_filename, split_file_base_path, start, end, stat):
	num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
	num_records_per_file = np.zeros(num_files)
	fileid_array = get_fileid_array(split_file_base_path, num_files, True)
	with open(input_filename) as infile:
		for line in islice(infile,start,end):
			data_element = line.strip().split(",")
			y = int(float(data_element[2]))
			place_id = int(float(data_element[5]))
			y0 = int(stat[place_id]/ct.Y_STEP_SIZE)
			#write_file(fileid_array, num_records_per_file, line, y0, num_files)
			fileid_array[y0].write(line)
			num_records_per_file[y0] = num_records_per_file[y0] + 1
	close_fileid_array(fileid_array)
	print sum(num_records_per_file), end-start
	ut.zip_and_delete(ut.get_split_files(split_file_base_path, num_files))

def split_file_pred(input_filename, split_file_base_path, start, end):
	num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
	num_records_per_file = np.zeros(num_files)
	record_id_to_file_id = []
	#print num_records_per_file
	fileid_array = get_fileid_array(split_file_base_path, num_files, False)
	with open(input_filename) as infile:
		for line in islice(infile,start,end):
			data_element = line.strip().split(",")
			y = int(float(data_element[2]))
			y0 = y/ct.Y_STEP_SIZE
			#rite_file(fileid_array, num_records_per_file, line, y0, num_files)
			fileid_array[y0].write(line)
			record_id_to_file_id.append(y0)
			num_records_per_file[y0] = num_records_per_file[y0] + 1
	close_fileid_array(fileid_array)
	
	#Save the record_id_to_file_id
	rifi_filename = split_file_base_path+ct.Y_BASED_SPLIT_REC_ID_TO_FIL_ID_NAME
	np.savetxt(rifi_filename, np.array(record_id_to_file_id), delimiter=",", fmt="%d");
	
	print sum(num_records_per_file), end-start
	ut.zip_and_delete(ut.get_split_files(split_file_base_path, num_files))

def split_file(input_filename, split_file_base_path, start, end, is_train, stat):
	#if(is_train):
	#	num_files = get_num_range()
	#else:
	num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
	num_records_per_file = np.zeros(num_files)
	#print num_records_per_file
	fileid_array = get_fileid_array(split_file_base_path, num_files, is_train)
	with open(input_filename) as infile:
		for line in islice(infile,start,end):
			data_element = line.strip().split(",")
			y = int(float(data_element[2]))
			if(is_train):
				#y0 = 3*(y/ct.Y_RANGE); 
				#y1 = 3*((y-ct.Y_STEP_SIZE)/ct.Y_RANGE)+1; 
				#y2 = 3*((y-2*ct.Y_STEP_SIZE)/ct.Y_RANGE)+2;
				#write_file(fileid_array, num_records_per_file, line, y0, num_files)
				#write_file(fileid_array, num_records_per_file, line, y1, num_files)
				#write_file(fileid_array, num_records_per_file, line, y2, num_files)
				place_id = int(float(data_element[5]))
				y0 = int(stat[place_id]/ct.Y_STEP_SIZE)
				write_file(fileid_array, num_records_per_file, line, y0, num_files)
			else:
				y0 = y/ct.Y_STEP_SIZE
				write_file(fileid_array, num_records_per_file, line, y0, num_files)
	close_fileid_array(fileid_array)
	print sum(num_records_per_file), end-start
	ut.zip_and_delete(ut.get_split_files(split_file_base_path, num_files))

def get_y_mean(input_filename):
	num_entries = ut.get_num_records(input_filename, 1)
	np_stat = np.empty(num_entries)
	with open(input_filename) as infile:
		for line in islice(infile,1,None):
			data_element = [int(float(x)) for x in line.strip().split(",")]
			np_stat[data_element[0]] = data_element[2]/data_element[7]
	return np_stat

total_records = ut.get_num_records(ct.TRAIN_ALL_INT_PATH, 1)
num_train_records = int(ct.TRAINING_PERCENT*total_records)
stat = get_y_mean(ct.STATS_PATH)
#split_file(ct.TRAIN_ALL_INT_PATH, ct.Y_BASED_SPLIT_TRAIN_PATH, 1, num_train_records, True, stat)
#split_file(ct.TRAIN_ALL_INT_PATH, ct.Y_BASED_SPLIT_EVAL_PATH, num_train_records+1, total_records, False)

#split_file_train(ct.TRAIN_ALL_INT_PATH, ct.Y_BASED_SPLIT_FULL_TRAIN_PATH, 1, total_records, stat)
#split_file_pred(ct.TEST_ALL_INT_PATH, ct.Y_BASED_SPLIT_TEST_PATH, 1, ut.get_num_records(ct.TEST_ALL_INT_PATH, 1))

split_file_train(ct.TRAIN_ALL_INT_PATH, ct.Y_BASED_SPLIT_TRAIN_PATH, 1, num_train_records, stat)
split_file_pred(ct.TRAIN_ALL_INT_PATH, ct.Y_BASED_SPLIT_EVAL_PATH, num_train_records+1, total_records)


