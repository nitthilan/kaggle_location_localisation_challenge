from itertools import islice
import matplotlib.pyplot as plt
import numpy as np

MEAN_STATISTICS = "../data/fb/train/mean_statistics.csv"
FILE_PATH = "/Volumes/movies/test_data/fb/modified.csv"
DISTRIBUTED_RECORD_PATH = "/Volumes/movies/test_data/fb/distributed_record/dr"


with open(MEAN_STATISTICS) as infile:
	stats = []; row_id = 0
	for line in islice(infile,1,None):
		data_element = line.strip().split(",")
		#data_element = [float(x) for x in data_element]
		record = [row_id, int(float(data_element[8]))]
		#print record
		stats.append(record)
		row_id = row_id + 1

np_stats = np.array(stats)
np_stats.view('i8,i8').sort(order=['f1'], axis=0)
print len(np_stats), np_stats.shape, np_stats.size
#print np_stats

#generating file_idx based on num entries
id_to_filename = {}
NUM_FILES = 100
for i in range(np_stats.shape[0]):
	id_to_filename[np_stats[i][0]] = (i%NUM_FILES, np_stats[i][1])
	#print id_to_filename[np_stats[i][0]]

for i in range(NUM_FILES):
	dr_filename = DISTRIBUTED_RECORD_PATH+str(i)+".csv"
	with open(dr_filename,"w") as outfile:
		outfile.write("row_id,x,y,accuracy,time,place_id\n")

print "Basic file creation done"
with open(FILE_PATH) as infile:
	row_id = 0
	for line in islice(infile,1,None):
		row_id = row_id + 1
		if(row_id%10000 == False):
			print row_id
		record = line.strip().split(",")
		pid = int(float(record[5]))
		dr_filename = DISTRIBUTED_RECORD_PATH+str(id_to_filename[pid][0])+".csv"
		#print id_to_filename[pid][0], line
		with open(dr_filename,"a") as outfile:
		 	outfile.write(line)



plt.plot(np_stats[::100,1],'ro')
plt.show()
