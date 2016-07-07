from itertools import islice
import gzip
import shutil
import os

def get_num_records(filename, start_line_num):
	num_lines = 0
	with open(filename) as infile:
		for line in islice(infile,start_line_num,None):
			num_lines = num_lines + 1
	return num_lines+1

def zip_and_delete(filename_list):
	for filename in filename_list:
		#print filename
		with open(filename, 'rb') as f_in, gzip.open(filename+".gz", 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)
		os.remove(filename)


# def get_file_list(directory):
# 	return [f for f in listdir(mypath) if isfile(join(directory, f))]

def get_split_files(filename_base, num_files):
	filename_list = []
	for idx in range(num_files):
		filename = filename_base+str(idx)+".csv"
		filename_list.append(filename)
	return filename_list


def get_xy_fileid_array(filename_base, num_x, num_y, isFileWrite):
	fileid_array = []
	for idx_x in range(num_x):
		fileid_array.append([])
		for idx_y in range(num_y):
			filename = filename_base+str(idx_x)+"_"+str(idx_y)+".csv"
			if(isFileWrite):
				fileid = open(filename, "w")
			else:
				fileid = open(filename)
			fileid_array[idx_x].append(fileid)
	return fileid_array

def close_xy_fileid_array(fileid_array, num_x, num_y):
	for idx_x in range(num_x):
		for idx_y in range(num_y):
			fileid_array[idx_x][idx_y].close()

def get_xy_split_files(filename_base, num_x, num_y):
	fileid_array = []
	for idx_x in range(num_x):
		for idx_y in range(num_y):
			filename = filename_base+str(idx_x)+"_"+str(idx_y)+".csv"
			fileid_array.append(filename)
	return fileid_array
