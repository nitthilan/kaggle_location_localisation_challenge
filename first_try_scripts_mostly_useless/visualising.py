from itertools import islice
import matplotlib.pyplot as plt
import numpy as np

MEAN_STATISTICS = "/Volumes/movies/test_data/fb/mean_statistics.csv"


with open(MEAN_STATISTICS) as infile:
	stats = []
	for line in islice(infile,1,None):
		data_element = line.strip().split(",")
		data_element = [float(x) for x in data_element]
		stats.append(data_element)

np_stats = np.array(stats)
print len(np_stats), np_stats.shape, np_stats.size

mean_x = (np_stats[:,0]/np_stats[:,8])
mean_xsq = np_stats[:,4]/np_stats[:,8]
sd_x = (mean_xsq - mean_x**2)
mean_y = (np_stats[:,1]/np_stats[:,8])
mean_ysq = np_stats[:,5]/np_stats[:,8]
sd_y = (mean_ysq - mean_y**2)
num_entries = np_stats[:,8]
sd_xy = (sd_x+sd_y)
mean_acc = np_stats[:,2]/np_stats[:,8]
time = np_stats[:,3]/np_stats[:,8]

# plt.scatter(mean_x, mean_y, s=num_entries)
# plt.show()
# plt.scatter(mean_x, mean_y, s=sd_xy)
# plt.show()
mng = plt.get_current_fig_manager()
#print dir(mng), mng.resize
#mng.resize(1280,800)
mng.full_screen_toggle()

#plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
#plt.scatter(mean_x[::5], mean_y[::5], s=mean_acc)
#plt.axis((0.0,1.0,0.0,1.0))
#plt.show()

# Individual histogram plots to observe pattern
# plt.hist(time,100)
# plt.show()
# plt.hist(mean_acc,100)
# plt.show()
# plt.hist(sd_x,100)
# plt.show()
# plt.hist(sd_y,100)
# plt.show()
# plt.hist(sd_xy,100)
# plt.show()

# Observing the number of entries distribution
plt.hist(num_entries, 100)
plt.show()
plt.plot(num_entries,'ro')
plt.show()
num_entries_sort = num_entries
num_entries_sort.sort()
plt.plot(num_entries_sort,'ro')
plt.show()

# Trying to understand the dependency between features by cross plotting 
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(mean_x, mean_y)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(mean_x, time)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(mean_x, mean_acc)
# plt.axis((0.0,1.0,0.0,.25))
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(mean_y, time)
# plt.axis((0.0,1.0,0.0,1.0))
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(mean_y, mean_acc)
# plt.axis((0.0,1.0,0.0,0.25))
# plt.show()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.scatter(time, mean_acc)
# plt.axis((0.0,1.0,0.0,0.25))
# plt.show()


# Not all points have equal samples. The distribution is in a S shape.
# Can we create a separate classification for each position separately
# That also means the number of points in the test would also be of simillar distribution
# There are very few points which have more points of distribution
# Number of points having around 100 points is the highest
# Try plotting the distribution of the points inside the large distribution and observe any pattern

# Based on histogram of individual features
# Y has a very uniform distribution
# X seems to have a slight drop around the extremes
# accuracy is peaking around 0.1 and dies down around 0.2 and trails to 1 (assuming peak at 1330)
# time has two peaks: peaking around 0.5 and a small peak around 0.75

# Cross corelation between features
# X and Y seems to be uniformly distributed very less correlation
# X/Y with Accuracy and time also seems to be less correlated and the 1D observation seems to replicate
# there seems to be a relation between accuracy and time. There are two distinct regions seen

# Multithreading to speedup things
# use KDTree for nearest Neighbour implementation
























