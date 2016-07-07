from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
import time


NUM_MODEL_FILE = 100
NUM_TEST_FILE = 30
PREDICTED_FILE = "./data/predicted_test/pt"
TOP_PREDICTED_FILE = "./data/predicted_test/tpt"
FID_LID_TO_LAID = "./data/fileid_localid_to_labelid"
ID_TO_IDX = "./data/id_to_idx"

def get_num_entries(filename):
	with open(filename) as infile:
		num_entires = 0
		for line in infile:
			num_entires = num_entires + 1
	return num_entires

def get_predicted_idx_value(testfileidx):
	predicted_value = np.zeros([total_entries, NUM_MODEL_FILE*3], dtype='f4')
	predicted_idx = np.zeros([total_entries, NUM_MODEL_FILE*3], dtype='i4')
	for modelfileidx in range(NUM_MODEL_FILE):
		infilename = PREDICTED_FILE+str(modelfileidx)+"_"+str(testfileidx)+".csv"
		with open(infilename) as infile:
			num_entires = 0
			for line in infile:
				record = [float(x) for x in line.strip().split(",")]
				predicted_value[num_entires][3*modelfileidx+0] = record[1]
				predicted_value[num_entires][3*modelfileidx+1] = record[3]
				predicted_value[num_entires][3*modelfileidx+2] = record[5]
				predicted_idx[num_entires][3*modelfileidx+0] = int(record[0])
				predicted_idx[num_entires][3*modelfileidx+1] = int(record[2])
				predicted_idx[num_entires][3*modelfileidx+2] = int(record[4])
				num_entires = num_entires + 1
			#print predicted_idx[num_entires-1], predicted_value[num_entires-1]
	return (predicted_value, predicted_idx)

def get_best_n_idx(predicted_value, predicted_idx, fileid_localid_to_labelid, id_to_idx):
	n = 3
	top_n = np.argsort(predicted_value)[:,:-n-1:-1]

	row_id = np.arange(len(predicted_value))
	col0 = predicted_idx[row_id, top_n[:,0]]
	col1 = predicted_idx[row_id, top_n[:,1]]
	col2 = predicted_idx[row_id, top_n[:,2]]

	fileid = (top_n/3).astype('i4')
	#print fileid.shape, col0.shape, fileid_localid_to_labelid[fileid[:,0], col0].shape

	pred0 = id_to_idx[fileid_localid_to_labelid[fileid[:,0], col0]]
	pred1 = id_to_idx[fileid_localid_to_labelid[fileid[:,1], col1]]
	pred2 = id_to_idx[fileid_localid_to_labelid[fileid[:,2], col2]]
	#print top_n.shape, predict.shape, col0.shape, col1.shape
	top_predicted = np.vstack((pred0, pred1, pred2)).transpose()
	return top_predicted

def get_id_to_idx():
	with open(ID_TO_IDX) as infile:
		id_to_idx = []
		for line in islice(infile,0,None):
			record = int(float(line.strip()))
			id_to_idx.append(record)
	return np.array(id_to_idx)

def get_fileid_localid_to_labelid():
	with open(FID_LID_TO_LAID) as infile:
		fileid_localid_to_labelid = np.zeros([100,1084], dtype='i4')
		lineid = 0
		for line in islice(infile,1,None):
			record = [int(float(x)) for x in line.strip().split(",")]
			for i in range(len(record)):
				fileid_localid_to_labelid[lineid, i] = record[i]
			lineid = lineid + 1
	return fileid_localid_to_labelid

def dump_pred(top_predicted, testfileidx):
	top_n_filename = TOP_PREDICTED_FILE+str(testfileidx)+".csv"
        np.savetxt(top_n_filename, top_predicted, delimiter=",", fmt="%d,%d,%d");


id_to_idx = get_id_to_idx()
fileid_localid_to_labelid = get_fileid_localid_to_labelid()
print id_to_idx.shape, fileid_localid_to_labelid.shape
for testfileidx in range(NUM_TEST_FILE):
	start = time.time()
	filename = PREDICTED_FILE+"0_"+str(testfileidx)+".csv"
	total_entries = get_num_entries(filename)
	predicted_value, predicted_idx = get_predicted_idx_value(testfileidx)
	top_predicted = get_best_n_idx(predicted_value, predicted_idx, fileid_localid_to_labelid, id_to_idx)
	#print top_predicted[total_entries-10], predicted_idx[total_entries-10], predicted_value[total_entries-10]
	dump_pred(top_predicted, testfileidx)
	print testfileidx, time.time() - start
	
	






















