import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
n_neighbors = 15

INPUT_FILE = "/Volumes/movies/test_data/fb/modified.csv"


def get_data(filename):
	feature_array = [];label_array = []
	with open(filename) as infile:
		for line in islice(infile,1,None):
			record = [float(x) for x in line.strip().split(",")]
			feature_array.append(record[1:5])
			label_array.append(int(record[5]))
	return (feature_array, label_array)



(feature_array, label_array) = get_data(INPUT_FILE)
print len(feature_array), len(feature_array[0]), len(label_array)
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(feature_array, label_array)



























