
from itertools import islice
import matplotlib.pyplot as plt
FILE_PATH = "/Volumes/movies/test_data/fb/train.csv"
#FILE_PATH = "../data/fb/train/xaa1"

NORMALISED_DATA_PATH = "/Volumes/movies/test_data/fb/modified.csv"
#NORMALISED_DATA_PATH = "../data/fb/train/xaa1_modified"
ID_TO_IDX = "/Volumes/movies/test_data/fb/id_to_idx"

# print len(idx), min(idx), max(idx)
# 108390 1000015801 9999932225

with open(FILE_PATH) as infile:
	#with open(NORMALISED_DATA_PATH,"w") as outfile:
		#outfile.write('row_id,x,y,accuracy,time,place_id\n')
		unique_entries = {}; num_entries = 0;
		id_to_idx = {}; idx = []
		for line in islice(infile,1,None):
			num_entries = num_entries + 1
			data_element = line.strip().split(",")
			#print data_element
			place_id = data_element[5]
			if(unique_entries.has_key(place_id)):
				unique_entries[place_id] = unique_entries[place_id] + 1;
			else:
				unique_entries[place_id] = 1;
				idx.append(place_id)
				id_to_idx[place_id] = len(idx)-1
			data_element[5] = id_to_idx[place_id]
			data_element[1] = float(data_element[1]) / 10.0
			data_element[2] = float(data_element[2]) / 10.0
			data_element[3] = float(data_element[3]) / 1033.0
			data_element[4] = float(data_element[4]) / 786239.0
			output_string = ",".join([str(x) for x in data_element])
			output_string = output_string + "\n"
			#outfile.write(output_string)
print len(idx), min(idx), max(idx)
print idx[0], idx[1], idx[2]
print id_to_idx[idx[0]], id_to_idx[idx[1]], id_to_idx[idx[2]]

with open(ID_TO_IDX, "w") as id_to_idx_file:
	for pid in idx:
		id_to_idx_file.write(str(pid))
		id_to_idx_file.write("\n")















