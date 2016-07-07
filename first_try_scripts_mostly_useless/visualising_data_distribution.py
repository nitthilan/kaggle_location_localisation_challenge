from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
REDUCED_DATA_PATH = "/Volumes/movies/test_data/fb/reduced_train/reduced0.csv"

unique_entries = {}
record_array = []
with open(REDUCED_DATA_PATH) as infile:
	for line in islice(infile,1,None):
		record = [float(x) for x in line.strip().split(',')]
		pid = int(record[5])
		record_array.append(record)
		if(unique_entries.has_key(pid) == False):
			unique_entries[pid] = []
		unique_entries[pid].append(record[1:5])
	num_entries = [len(unique_entries[key]) for key in unique_entries]
	print min(num_entries), max(num_entries)
	np_record_array = np.array(record_array)
	#print np_record_array.shape, np_record_array[1:10,3], np_record_array[1:10,4]


def plot_subplot(records):
	np_records = np.array(records)
	X = np_records[:,0]
	Y = np_records[:,1]
	A = np_records[:,2]
	T = np_records[:,3]
	f, axarr = plt.subplots(2, 3)
	f.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0.1, wspace=0.1)
	axarr[0, 0].axis((0.0,1.0,0.0,1.0));axarr[0, 1].axis((0.0,1.0,0.0,0.4));axarr[0, 2].axis((0.0,1.0,0.0,1.0))
	axarr[1, 0].axis((0.0,1.0,0.0,0.4));axarr[1, 1].axis((0.0,1.0,0.0,1.0));axarr[1, 2].axis((0.0,0.4,0.0,1.0))
	axarr[0, 0].scatter(X,Y)
	axarr[0, 1].scatter(X,A)
	axarr[0, 2].scatter(X,T)
	axarr[1, 0].scatter(Y,A)
	axarr[1, 1].scatter(Y,T)
	axarr[1, 2].scatter(A,T)
	plt.show()

num_plots = 0;MAX_NUM_PLOTS = 10
for key in unique_entries:
	if(len(unique_entries[key]) > 20 and len(unique_entries[key]) < 40):
		num_plots = num_plots + 1
		print num_plots,len(unique_entries[key])
		plot_subplot(unique_entries[key])
	#print num_plots;
	if num_plots == MAX_NUM_PLOTS:
		break

# plt.hist(num_entries)
# plt.show()
# plt.plot(num_entries,"ro")
# plt.show()



# DF = 100
# X = np_record_array[::DF,1]
# Y = np_record_array[::DF,2]
# A = np_record_array[::DF,3]
# T = np_record_array[::DF,4]
# I = np_record_array[::DF,5].astype(int)
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(X, Y, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(X, A, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(X, T, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(Y, A, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(Y, T, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.scatter(A, T, c=I, s=(100.0*I/108390), alpha=0.5)
# plt.show()

# General understanding:
# Find range of each field (min, max, mean, std deviation)
# Find num entries in each category
# Find num unique categories
# Normalise the input using the min and max and map it to the scaled values
# Try getting single instance distribution using histogram
# Try getting inter feature relation by using 2D scatter plots