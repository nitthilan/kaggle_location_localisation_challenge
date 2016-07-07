from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import time


TEST_PATH = "./data/test.csv"
DR_FILENAME = "./data/fb/distributed_record/dr0.csv"
TEST_NORMALISE_SPLIT_FILENAME = "./data/test_normalised_split/tns"

NUM_RECORDS_PER_FILE = 291180
TOTAL_NUM_RECORDS = 8607230



# Total num of records 8607230, 29118021
# with open(TEST_PATH) as infile:
# 	num_test = 0;
# 	for line in islice(infile,1,None):
# 		num_test = num_test + 1
# 	print line, num_test

# understanding the test data
# print xtotal/num_entries, xmax, xmin
# print ytotal/num_entries, ymax, ymin
# print atotal/num_entries, amax, amin
# print ttotal/num_entries, tmax, tmin
# 32711.2475175 65536.0 0.0
# 32811.4416188 65536.0 0.0
# 92.6520762196 1026.0 1.0
# 890463.661617 1006589.0 786242.0


def get_train_data(filename):
	with open(DR_FILENAME) as infile:
		num_entires = 0
		for line in islice(infile,1,None):
			num_entires = num_entires + 1
		feature = np.empty([num_entires, 4], dtype='i4')
		label = np.empty([num_entires, 1], dtype='i4')
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

def min_max_total(xmin, xmax, xtotal, x):
	x = float(x)
	if(x < xmin): xmin = x
	if(x > xmax): xmax = x
	xtotal = xtotal + x
	return (xmin, xmax, xtotal)

def normalising_test():
	outfile_id = 0; num_records = 0; fileout_data = ""
	with open(TEST_PATH) as infile:
		xmin = 10000.0; xmax = 0.0; xtotal = 0
		ymin = 10000.0; ymax = 0.0; ytotal = 0
		amin = 100000000.0; amax = 0.0; atotal = 0
		tmin = 100000000.0; tmax = 0.0; ttotal = 0
		for line in islice(infile,1,None):
			data_element = line.strip().split(",")
			data_element[1] = int(float(data_element[1])*65536 / 10.0)
			data_element[2] = int(float(data_element[2])*65536 / 10.0)
			data_element[3] = int(float(data_element[3]))
			data_element[4] = int(float(data_element[4]))
			#(xmin, xmax, xtotal) = min_max_total(xmin, xmax, xtotal, data_element[1])
			#(ymin, ymax, ytotal) = min_max_total(ymin, ymax, ytotal, data_element[2])
			#(amin, amax, atotal) = min_max_total(amin, amax, atotal, data_element[3])
			#(tmin, tmax, ttotal) = min_max_total(tmin, tmax, ttotal, data_element[4])
			
			output_string = ",".join([str(x) for x in data_element])
			output_string = output_string + "\n"
			fileout_data = fileout_data + output_string
			num_records = num_records + 1
			
			#print output_string
			if (num_records % NUM_RECORDS_PER_FILE == False or num_records == TOTAL_NUM_RECORDS):
				outfile_name = TEST_NORMALISE_SPLIT_FILENAME+str(outfile_id)+".csv"
				print outfile_name
				with open(outfile_name, "w") as outfile:
					outfile.write("row_id,x,y,accuracy,time\n");
					outfile.write(fileout_data)
				outfile_id = outfile_id + 1
				fileout_data = ""
		#print xtotal/num_records, xmax, xmin
		#print ytotal/num_records, ymax, ymin
		#print atotal/num_records, amax, amin
		#print ttotal/num_records, tmax, tmin
normalising_test()


