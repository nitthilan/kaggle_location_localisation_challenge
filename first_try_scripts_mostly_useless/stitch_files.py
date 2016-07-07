

# Sample output
# row_id,place_id
# 0,3073560757 9004412889 5652080691

TOP_PREDICTED_FILE = "./data/predicted_test/tpt"
MERGED_FILE = "./data/predicted_test/top_3_pred.csv"

NUM_FILES = 30


with open(MERGED_FILE, "w") as outfile:
	outfile.write("row_id,place_id\n");
	row_id = 0
	for i in range(NUM_FILES):
		infilename = TOP_PREDICTED_FILE+str(i)+".csv"
		with open(infilename) as infile:
			for line in infile:
				split_record = line.strip().split(",")
				join_record = " ".join(split_record)
				outfile.write(str(row_id)+","+join_record+"\n");
				row_id = row_id + 1


