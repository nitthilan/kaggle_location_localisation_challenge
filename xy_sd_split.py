#!/usr/bin/python
from itertools import islice
import numpy as np
import zipfile
import gzip
import shutil
import os

from os import listdir
from os.path import isfile, join

import constants as ct
import utils as ut


def get_xy_mean_sd(input_filename, x_rng, y_rng, num_sd):
	num_entries = ut.get_num_records(input_filename, 1)
	max_x_end = int((ct.MAX_XY_VALUE-1)/x_rng)
	max_y_end = int((ct.MAX_XY_VALUE-1)/y_rng)
	print max_x_end, max_y_end, ((ct.MAX_XY_VALUE-1)/x_rng)+1, ((ct.MAX_XY_VALUE-1)/y_rng)+1, num_entries
	# x_start, x_end, y_start, y_end, num_entries
	np_stat ={}#np.empty((num_entries,5), dtype=int)
	with open(input_filename) as infile:
		for line in islice(infile,1,None):
			data_element = [(float(x)) for x in line.strip().split(",")]
			place_id = int(data_element[0])
			num_entries = data_element[7];
			x_mean = data_element[1]/num_entries; y_mean = data_element[2]/num_entries
			
			x_sqr_mean = data_element[4]/num_entries; y_sqr_mean = data_element[5]/num_entries
			
			x_sd = np.sqrt(x_sqr_mean - x_mean**2); y_sd = np.sqrt(y_sqr_mean - y_mean**2)

			x_start = int((x_mean - num_sd*x_sd)/x_rng)
			if(x_start < 0):
				x_start = 0

			#print data_element, y_sqr_mean, y_mean, y_sd, (y_mean - num_sd*y_sd)
			y_start = int((y_mean - num_sd*y_sd)/y_rng)
			if(y_start < 0):
				y_start = 0
			x_end = int((x_mean + num_sd*x_sd)/x_rng)
			if(x_end > max_x_end):
				x_end = max_x_end
			y_end = int((y_mean + num_sd*y_sd)/y_rng)
			if(y_end > max_y_end):
				y_end = max_y_end
			if(not np_stat.has_key(place_id)):
				np_stat[place_id] = [0,0,0,0,0]
			np_stat[place_id][0] = x_start
			np_stat[place_id][1] = x_end
			np_stat[place_id][2] = y_start
			np_stat[place_id][3] = y_end
			np_stat[place_id][4] = num_entries
	return np_stat

def split_file_train(input_filename, split_file_base_path, x_rng, y_rng, start, end, stat):
	num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1

	fileid_array = ut.get_xy_fileid_array(split_file_base_path, num_x, num_y, True);

	num_records_per_file = np.zeros((num_x, num_y))
	with open(input_filename) as infile:
		for line in islice(infile,start,end):
			data_element = line.strip().split(",")
			place_id = int(float(data_element[5]))
			for idx_x in range(stat[place_id][0], stat[place_id][1]+1):
				for idx_y in range(stat[place_id][2], stat[place_id][3]+1):
					#print idx_y, idx_x, len(fileid_array), len(fileid_array[0])
					fileid_array[idx_x][idx_y].write(line)
					num_records_per_file[idx_x][idx_y] = num_records_per_file[idx_x][idx_y] + 1
	ut.close_xy_fileid_array(fileid_array, num_x, num_y)
	print sum(sum(num_records_per_file)), end-start
	#ut.zip_and_delete(ut.get_xy_split_files(split_file_base_path, num_x, num_y))


def split_file_pred(input_filename, split_file_base_path, x_rng, y_rng, start, end):
	num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1
	print num_x, num_y

	fileid_array = ut.get_xy_fileid_array(split_file_base_path, num_x, num_y, True);

	num_records_per_file = np.zeros((num_x, num_y))

	record_id_to_file_id = []
	#print num_records_per_file
	with open(input_filename) as infile:
		for line in islice(infile,start,end):
			data_element = line.strip().split(",")
			y0 = int(float(data_element[2])/y_rng)
			x0 = int(float(data_element[1])/x_rng)
			if(x0 > num_x-1):
				x0 = num_x-1
			if(y0 > num_y-1):
				y0 = num_y-1
			#print x0, y0

			#rite_file(fileid_array, num_records_per_file, line, y0, num_files)
			fileid_array[x0][y0].write(line)
			record_id_to_file_id.append([x0, y0])
			num_records_per_file[x0][y0] = num_records_per_file[x0][y0] + 1
	ut.close_xy_fileid_array(fileid_array, num_x, num_y)
	
	#Save the record_id_to_file_id
	rifi_filename = split_file_base_path+ct.REC_ID_TO_FIL_ID_NAME
	np.savetxt(rifi_filename, np.array(record_id_to_file_id), delimiter=",", fmt="%d");
	
	print sum(num_records_per_file), end-start
	#ut.zip_and_delete(ut.get_xy_split_files(split_file_base_path, num_x, num_y))


# total_records = ut.get_num_records(ct.TRAIN_ALL_INT_PATH, 1)
# num_train_records = int(ct.TRAINING_PERCENT*total_records)
# stat = get_xy_mean_sd(ct.STATS_PATH, ct.X_RNG, ct.Y_RNG, ct.NUM_SD)
#split_file_train(ct.TRAIN_ALL_INT_PATH, ct.XY_SD_SP_TR80_PATH, ct.X_RNG, ct.Y_RNG, 1, num_train_records, stat)
#split_file_pred(ct.TRAIN_ALL_INT_PATH,ct.XY_SD_SP_EVAL_PATH, ct.X_RNG, ct.Y_RNG, num_train_records, total_records)


# stat = get_xy_mean_sd(ct.SPLIT_STATS_PATH, ct.X_RNG, ct.Y_RNG, ct.NUM_SD)
# split_file_train(ct.SPLIT_TRAIN_ALL_INT_PATH, ct.XY_SD_SP_TR_TS_2SD_PATH, ct.X_RNG, ct.Y_RNG, 1, ut.get_num_records(ct.SPLIT_TRAIN_ALL_INT_PATH, 1), stat)
# split_file_pred(ct.SPLIT_EVAL_ALL_INT_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PATH, ct.X_RNG, ct.Y_RNG, 1, ut.get_num_records(ct.SPLIT_EVAL_ALL_INT_PATH, 1))


# split_file_train(ct.TRAIN_ALL_INT_PATH, ct.XY_SD_SP_FUTR_PATH, ct.X_RNG, ct.Y_RNG, 1, total_records, stat)
# split_file_pred(ct.TEST_ALL_INT_PATH, ct.XY_SD_SP_TEST_PATH, ct.X_RNG, ct.Y_RNG, 1, ut.get_num_records(ct.TEST_ALL_INT_PATH, 1))

