
from sklearn import preprocessing
from itertools import islice


FILE_PATH = "../data/fb/distributed_record/dr"
OUTFILE = "../data/fb/distributed_record_local_label/drll"
FID_LID_2_LID = "../data/fb/fileid_localid_to_labelid"

NUM_FILE = 100

fileid_localid_to_labelid = []
for fileid in range(NUM_FILE):
	infilename = FILE_PATH+str(fileid)+".csv"
	print infilename
	label_array = []
	record_array = []
	with open(infilename) as infile:
		for line in islice(infile,1,None):
			record = line.strip().split(',')
			label_array.append(int(float(record[5])))
			record_array.append(line)
	
	le = preprocessing.LabelEncoder()
	transform_label = le.fit_transform(label_array)
	fileid_localid_to_labelid.append(le.classes_)
	#print ','.join(map(str, le.classes_))+"\n"
	#print le.classes_

	outfilename = OUTFILE+str(fileid)+".csv"
	print outfilename
	with open(outfilename,"w") as outfile:
		outfile.write("row_id,x,y,accuracy,time,place_id\n");
		for idx in range(len(label_array)):
			record = record_array[idx].strip().split(',')
			record[5] = str(transform_label[idx])
			outfile.write(','.join(record)+"\n")

with open(FID_LID_2_LID,"w") as outfile:
	outfile.write("2D row is fileid, column is labelid inside that file\n")
	for fileid in range(NUM_FILE):
		outfile.write(','.join(map(str, fileid_localid_to_labelid[fileid]))+"\n")


