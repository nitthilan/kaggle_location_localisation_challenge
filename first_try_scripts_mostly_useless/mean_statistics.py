from itertools import islice
import matplotlib.pyplot as plt
FILE_PATH = "/Volumes/movies/test_data/fb/modified.csv"
#FILE_PATH = "../data/fb/train/xaa1_modified"

ID_TO_IDX = "/Volumes/movies/test_data/fb/id_to_idx"
#ID_TO_IDX = "../data/fb/train/id_to_idx"

#MEAN
MEAN_STATISTICS = "/Volumes/movies/test_data/fb/mean_statistics.csv"
#MEAN_STATISTICS = "../data/fb/train/mean_statistics"


with open(FILE_PATH) as infile:
	unique_entries = {}; idx = []
	for line in islice(infile,1,None):
		data_element = line.strip().split(",")
		place_id = data_element[5]
		if(unique_entries.has_key(place_id) == False):
			# X, Y, Accuracy, Time, NumEntries
			unique_entries[place_id] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
			idx.append(place_id)
		num_entries = unique_entries[place_id][4]
		for i in range(4):
			value = float(data_element[i+1])
			unique_entries[place_id][i] = unique_entries[place_id][i] + value
			unique_entries[place_id][4+i] = unique_entries[place_id][4+i] + value*value
		unique_entries[place_id][8] = unique_entries[place_id][8] + 1
		

for i in range(10):
	print ",".join(str(x) for x in unique_entries[idx[i]])

print len(idx)

with open(MEAN_STATISTICS, "w") as meanfile:
	meanfile.write("x_sum, y_sum, acc_sum, time_sum, xsq_sum, ysq_sum, accsq_sum, timesq_sum, num_entries\n")
	for pid in idx:
		output_string = ",".join(str(x) for x in unique_entries[pid])
		output_string = output_string + "\n"
		meanfile.write(output_string)
















