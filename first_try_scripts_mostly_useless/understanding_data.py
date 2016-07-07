from itertools import islice
import matplotlib.pyplot as plt

# List of things to pre-process:
# =============================
# - Number of samples in each id
# - Average value for x,y,accuracy,time,place_id
# - number of place_id

FOLDER_PATH = "../data/fb/train/"
#FILE_PATH = "../data/fb/train/xaa1"
FILE_PATH = "/Volumes/movies/test_data/fb/train.csv"

NORMALISED_DATA_PATH = "/Volumes/movies/test_data/fb/modified.csv"

#https://www.quora.com/What-are-the-advantages-of-different-classification-algorithms

#Statistics of the data
# Num unique entries: 108390 
# print num_entries, len(unique_entries.keys()), min(unique_entries.values()), max(unique_entries.values()), sum(unique_entries.values())
# print xtotal/num_entries, xmax, xmin
# print ytotal/num_entries, ymax, ymin
# print atotal/num_entries, amax, amin
# print ttotal/num_entries, tmax, tmin
# 29118021 108390 1 1849 29118021
# 4.99976983066 10.0 0.0
# 5.00181392855 10.0 0.0
# 82.8491249457 1033.0 1.0
# 417010.364723 786239.0 1.0

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

def min_max_total(xmin, xmax, xtotal, x):
	x = float(x)
	if(x < xmin): xmin = x
	if(x > xmax): xmax = x
	xtotal = xtotal + x
	return (xmin, xmax, xtotal)


#Reading large files
#http://stackoverflow.com/questions/6475328/read-large-text-files-in-python-line-by-line-without-loading-it-in-to-memory
with open(FILE_PATH) as infile:
	unique_entries = {}; num_entries = 0; unique_entries = {}
	xmin = 10000.0; xmax = 0.0; xtotal = 0
	ymin = 10000.0; ymax = 0.0; ytotal = 0
	amin = 100000000.0; amax = 0.0; atotal = 0
	tmin = 100000000.0; tmax = 0.0; ttotal = 0
	for line in islice(infile,1,None):
		num_entries = num_entries + 1
		data_element = line.strip().split(",")
		#print data_element
		place_id = data_element[5]
		if(unique_entries.has_key(place_id)):
			unique_entries[place_id] = unique_entries[place_id] + 1;
		else:
			unique_entries[place_id] = 1;
		(xmin, xmax, xtotal) = min_max_total(xmin, xmax, xtotal, data_element[1])
		(ymin, ymax, ytotal) = min_max_total(ymin, ymax, ytotal, data_element[2])
		(amin, amax, atotal) = min_max_total(amin, amax, atotal, data_element[3])
		(tmin, tmax, ttotal) = min_max_total(tmin, tmax, ttotal, data_element[4])

		

	print num_entries, len(unique_entries.keys()), min(unique_entries.values()), max(unique_entries.values()), sum(unique_entries.values())
	print xtotal/num_entries, xmax, xmin
	print ytotal/num_entries, ymax, ymin
	print atotal/num_entries, amax, amin
	print ttotal/num_entries, tmax, tmin
	num_entries_distribution = unique_entries.values()
	num_entries_distribution.sort();
	#print num_entries_distribution
	plt.plot(num_entries_distribution)
	plt.show()


   