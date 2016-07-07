import zipfile
import os

NUM_MODEL_FILE = 100
NUM_TEST_FILE = 30
PREDICTED_FILE = "./data/predicted_test/pt"

for i in range(NUM_MODEL_FILE):
	for j in range(NUM_TEST_FILE):
		infile = PREDICTED_FILE+str(i)+"_"+str(j)+".csv"
		outzipfile = PREDICTED_FILE+str(i)+"_"+str(j)+".zip"
		with zipfile.ZipFile(outzipfile, 'w', zipfile.ZIP_DEFLATED) as myzip:
			myzip.write(infile)
			os.remove(infile)

