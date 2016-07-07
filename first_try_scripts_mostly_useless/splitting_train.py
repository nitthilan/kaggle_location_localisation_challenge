from itertools import islice
import matplotlib.pyplot as plt
NORMALISED_DATA_PATH = "/Volumes/movies/test_data/fb/modified.csv"
REDUCED_DATA_PATH = "/Volumes/movies/test_data/fb/reduced_train/reduced"
NUM_FILES = 100

with open(NORMALISED_DATA_PATH) as infile:
	for filenum in range(NUM_FILES):
		reduced_file_name = REDUCED_DATA_PATH+str(filenum)+".csv"
		with open(reduced_file_name, "w") as outfile:
			outfile.write("row_id,x,y,accuracy,time,place_id\n")
			infile.seek(0)
			for line in islice(infile,1+filenum,None,NUM_FILES):
				#print reduced_file_name,filenum, line, 
				outfile.write(line)


