#!/usr/bin/python
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn import preprocessing
import time
from sklearn.externals import joblib
import constants as ct
import gzip
import utils as ut

def get_data_old(filename, is_train_data):
    NUM_FEATURES = 7
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        #print num_entires
        feature = np.empty([num_entires, NUM_FEATURES], dtype='i4')
        label = np.empty([num_entires, 1], dtype='i4').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
                data_element = line.strip().split(",")
                data_element = [int(float(x)) for x in data_element]
                feature[row_id][0] = data_element[1]
                feature[row_id][1] = data_element[2]
                feature[row_id][2] = np.log2(data_element[3])*100

                # feature[row_id][0] = int(data_element[1]*10000)
                # feature[row_id][1] = int(data_element[2]*10000)
                # feature[row_id][2] = int(data_element[3]*1033)
                nH = int(data_element[4]/60.0) # Num Hours - Cannot be used as feature
                Minute = data_element[4]%60.0 # May not be useful
                HiD24 = nH%24 #Hour in Day (0-24)
                HiD12 = (nH+12)%24 #Hour in Day (12-12)
                nD = int(nH/24) #numDays - Cannot be used as feature
                DiW = nD%7 #Day in Week
                DiM = nD%30.5 #Day in Month
                DiBW = nD%14 #Day in Bi-Week
                DiFW = nD%28 #Day in Four-Week
                DiY = nD%365 #Day in Year

                feature[row_id][3] = HiD24
                feature[row_id][4] = DiW
                # feature[row_id][5] = DiBW
                # feature[row_id][6] = int(DiFW/7)
                feature[row_id][5] = DiM
                feature[row_id][6] = nH
                # feature[row_id][8] = DiY
                # if(NUM_FEATURES >= 4):
                #     T = data_element[4]/(24*60.0)
                #     #T = data_element[4]*786239.0/(24*60.0)
                #     D = int(T)
                #     H = int(24*(T-D))
                #     feature[row_id][3] = H
                # if(NUM_FEATURES >= 5):
                #     feature[row_id][4] = (D%7)
                #     feature[row_id][5] = int(D/7)
                #     feature[row_id][6] = int(D/30)
                #     feature[row_id][7] = 60*(24*(T - D) - H)
		if(is_train_data):
                	label[row_id] = data_element[5]
                #print feature[row_id], label[row_id]
                row_id = row_id + 1
    #print feature.shape, label.shape
    if(is_train_data):
    	return (feature, label)
    else:
	   return feature

# def get_data(filename, is_train_data, offset_factor=0.0):
#     NUM_FEATURES = 7
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         feature = np.empty([num_entires, NUM_FEATURES], dtype='i4')
#         label = np.empty([num_entires, 1], dtype='i4').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [int(float(x)) for x in data_element]
#             feature[row_id][0] = data_element[1]#/100000.0
#             feature[row_id][1] = data_element[2]#/100000.0
#             feature[row_id][2] = np.log2(data_element[3])*64 #/1033.0 #np.log()*100
#             #feature[row_id][3] = (data_element[4]%(60*24))
#             mod_value = (60*24.0); offset = offset_factor*mod_value
#             time_offset = ((data_element[4]+offset) % mod_value)/mod_value
#             feature[row_id][3] = (np.sin(2*np.pi*time_offset)+1)*mod_value/12 #10,11,13
#             feature[row_id][4] = (np.cos(2*np.pi*time_offset)+1)*mod_value/12
#             num_days = (data_element[4]//mod_value)
#             feature[row_id][5] = (num_days%7)*27 #25,30 # There seems to be a influence by the number of weeks
#             feature[row_id][6] = (num_days//3) #2,4,5 # This feature gives importance to the placeid which are logged in recently
#             # feature[row_id][7] = (num_days%30)*0.5 # There seems to be no influence from month
#             # Since nearest neighbor seems to depend on distance, scale the inputs to meatch the dimension or importance of each feature
#             # If adding a feature does not change the output metric then the feature is not given enough importance to have any significant impact
#             # So try out values in the order of 1,10,100 till they start having impact. Once they start having impact, fine tune it by increasing or decreasing
#             # to find out whether the output metric (mean) increases or decreases proportionaly or inversely. Then fine tune in that direction to get the apporpriate weights

#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature
# def calculate_distance(distances):
#     distances[distances < .0001] = .0001
#     return distances ** -2

# def get_data(filename, is_train_data):
#     sf0=17.9548; sf1= 38.7297; sf2=0.6362; sf3=0.4655; sf4=0.077439; sf5=0.18514; sf6=0.46963; sf7=0.15897; sf8=0.1; sf9=0.1; cte=8.4592
#     #scale_factor = [22, 52, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
#     scale_factor = [sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9]

#     NUM_FEATURES = 10
#     with open(filename) as infile:
#         num_entires = 0
#         for line in islice(infile,0,None):
#             num_entires = num_entires + 1
        
#         record_array = np.empty([num_entires, 4], dtype="f8")
#         feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
#         label = np.empty([num_entires, 1], dtype='f8').ravel()
#         row_id = 0
#         infile.seek(0)
#         for line in islice(infile,0,None):
#             data_element = line.strip().split(",")
#             data_element = [(float(x)) for x in data_element]
#             record_array[row_id] = data_element[1:5]
#             if(is_train_data):
#                 label[row_id] = data_element[5]
#             row_id = row_id + 1
        
#         feature[:,0] = scale_factor[0]*(record_array[:,0]/10000.0)
#         feature[:,1] = scale_factor[1]*(record_array[:,1]/10000.0)
#         feature[:,2] = scale_factor[2]*np.log10(record_array[:,2])

#         time = record_array[:,3]
#         #minute = 2*np.pi*((time//5)%288)/288
#         minute = 2*np.pi*((time)%1440)/1440
#         feature[:,3] = scale_factor[3]*(np.sin(minute)+1).round(4)
#         feature[:,4] = scale_factor[3]*(np.cos(minute)+1).round(4)
        
#         day = 2*np.pi*((time//1440)%365)/365
#         feature[:,5] = scale_factor[5]*(np.sin(day)+1).round(4)
#         feature[:,6] = scale_factor[5]*(np.cos(day)+1).round(4)

#         #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
#         #feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001
#         feature[:,7] = scale_factor[6]*(time//525600)*1.0#*0.1#*0.00001
#         #print feature[:,9]

#         weekday = 2*np.pi*((time//1440)%7)/7
#         feature[:,8] = scale_factor[7]*(np.sin(weekday)+1).round(4)
#         feature[:,9] = scale_factor[7]*(np.cos(weekday)+1).round(4)

#     if(is_train_data):
#         return (feature[:row_id], label[:row_id])
#     else:
#         return feature

def scale_features_onlyNN(feature):
    # {'max_val': 0.58152332049636291, 'max_params': {'sf10': 0.46073844421128463, 'sf7': 0.49775812013822063, 'sf2': 0.55002881263155412, 'sf8': 0.18975171704703428, 'sf9': 0.09772323821239208}}
    # sf0=17.9548, sf1= 38.7297, sf2=0.5500, sf3=0.4655, sf4=0.18514, sf5=0.46963, sf6=0.15897; sf7=0.4978; sf8=0.4978; sf9=0.4978; sf10=0.4607 ;cte=8.4592, gpv=-2.8465
    sf0=17.9548; sf1= 38.7297; sf2=0.5500; sf3=0.4655; sf4=0.18514; sf5=0.46963; sf6=0.15897; sf7=0.4978; sf8=0.4978; sf9=0.4978; sf10=0.4607;
    #scale_factor = [22, 52, 0.6, 0.32535, 0.0056515, 0.2670, 0.051985]
    #scale_factor = [sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9]
    scale_factor = [sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9, sf10]

    # location features
    feature[:,0] *= scale_factor[0]
    feature[:,1] *= scale_factor[1]
    #Accuracy log
    feature[:,2] *= scale_factor[2]
    #Time features
    feature[:,3] *= scale_factor[3]
    feature[:,4] *= scale_factor[3]
    feature[:,5] *= scale_factor[4]
    feature[:,6] *= scale_factor[4]
    feature[:,7] *= scale_factor[5]
    feature[:,8] *= scale_factor[6]
    feature[:,9] *= scale_factor[6]
    #Accuracy
    feature[:,10] *= scale_factor[7]
    feature[:,11] *= scale_factor[8]
    feature[:,12] *= scale_factor[9]
    feature[:,13] *= scale_factor[10]
    return feature

def get_data(filename, is_train_data):
    NUM_FEATURES = 14
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,0,None):
            num_entires = num_entires + 1
        
        record_array = np.empty([num_entires, 4], dtype="f8")
        feature = np.zeros([num_entires, NUM_FEATURES], dtype='f8')
        label = np.empty([num_entires, 1], dtype='f8').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,0,None):
            data_element = line.strip().split(",")
            data_element = [(float(x)) for x in data_element]
            record_array[row_id] = data_element[1:5]
            if(is_train_data):
                label[row_id] = data_element[5]
            row_id = row_id + 1
        
        feature[:,0] = (record_array[:,0]/10000.0)
        feature[:,1] = (record_array[:,1]/10000.0)
        feature[:,2] = np.log10(record_array[:,2])

        time = record_array[:,3]
        #minute = 2*np.pi*((time//5)%288)/288
        minute = 2*np.pi*((time)%1440)/1440
        feature[:,3] = (np.sin(minute)+1).round(4)
        feature[:,4] = (np.cos(minute)+1).round(4)
        
        day = 2*np.pi*((time//1440)%365)/365
        feature[:,5] = (np.sin(day)+1).round(4)
        feature[:,6] = (np.cos(day)+1).round(4)

        #feature[row_id][9] = scale_factor[6]*(data_element[4]//3)*0.00001
        #feature[:,9] = scale_factor[6]*(time//(30*1440))#*0.1#*0.00001
        feature[:,7] = (time//525600)*1.0#*0.1#*0.00001
        #print feature[:,9]

        weekday = 2*np.pi*((time//1440)%7)/7
        feature[:,8] = (np.sin(weekday)+1).round(4)
        feature[:,9] = (np.cos(weekday)+1).round(4)

        accuracy = record_array[:,2]
        mask_0_45 = accuracy<45
        mask_45_100 = (accuracy >= 45) & (accuracy < 100)
        mask_100_200 = (accuracy >= 100) & (accuracy < 200)
        mask_200_1033 = (accuracy >= 200)
        feature[mask_0_45,10] = (accuracy[mask_0_45]/45)
        feature[mask_45_100,11] = (accuracy[mask_45_100]-45)/55
        feature[mask_100_200,12] = (accuracy[mask_100_200]-100)/100
        feature[mask_200_1033,13] = (accuracy[mask_200_1033]-200)/1024

        feature = scale_features_onlyNN(feature)

    if(is_train_data):
        return (feature[:row_id], label[:row_id])
    else:
        return feature

def calculate_distance(distances):
    #distances[distances < .0001] = .0001
    return distances ** -1#-2.8465#-2.75 # -2.22


def get_trained_classifier(train_filename):
        start_time = time.time()
        feature, label = get_data(train_filename, True)
        train_file_read_time = time.time() - start_time
        #print feature.shape, label.shape

        start_time = time.time()
        cte=8.4523#8.4592 #cte = 7.65 #5.8
        n_neighbors = int((label.size ** 0.5) / cte)
        #classifier = RandomForestClassifier(NUM_ESTIMATOR, max_features=4, max_depth=MAX_DEPTH, n_jobs=16)
        #classifier = RandomForestClassifier(n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1)
        #classifier = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=calculate_distance, metric='manhattan', n_jobs = -1)

        classifier.fit(feature, label)
        fit_time = time.time() - start_time
        print train_file_read_time, fit_time
        return classifier

def get_prediction(classifier, test_filename):
        start_time = time.time()
        feature = get_data(test_filename, False)
        test_file_read_time = time.time() - start_time

        start_time = time.time()
        predict = classifier.predict_proba(feature)
        predict_time = time.time() - start_time

        # argnmax is very slow compared to argmax. So better to replace value with negative value than Nan.
        start_time = time.time()
        top_0 = np.argmax(predict, axis=1)
        
        classes = classifier.classes_ 
        row_id = np.arange(len(predict))
        
        top_class_0 = classes[top_0]
        col_0 = predict[row_id, top_0]
        predict[row_id, top_0] = -1.0
        top_1 = np.argmax(predict, axis=1)
        top_class_1 = classes[top_1]
        col_1 = predict[row_id, top_1]
        predict[row_id, top_1] = -1.0
        top_2 = np.argmax(predict, axis=1)
        top_class_2 = classes[top_2]
        col_2 = predict[row_id, top_2]
        #argmax_top_n = np.vstack((top_0, top_1, top_2)).transpose()
        #argmax_predict_value = np.vstack((col_0, col_1, col_2)).transpose()
        top_n_predicted = np.vstack((top_class_0, col_0, top_class_1, col_1, top_class_2, col_2)).transpose()
        argmax_time = time.time() - start_time

        #print argmax_time, (argmax_predict_value == top_n).all(), (predict_value == argmax_predict_value).all()
        #print argmax_top_n[5], top_n[5], argmax_predict_value[5], predict_value[5]

        print test_file_read_time, predict_time, argmax_time #argsort_time, stack_array_time

        #return top_n, predict_value
        #return argmax_top_n, argmax_predict_value
        return top_n_predicted

def dump_predicted_n(top_n_predicted, top_n_filename):
        start_time = time.time()
        #predicted_filename = PREDICTED_TEST_FILENAME+"_pv_"+str(train_idx)+"_"+str(test_idx)+".csv"
        np.savetxt(top_n_filename, top_n_predicted, delimiter=",", fmt="%d,%0.2f,%d,%0.2f,%d,%0.2f");
        #np.savetxt(predicted_filename, predicted_value, delimiter=",", fmt="%0.6f");
        dump_time = time.time() - start_time
        print dump_time

def get_predicted_n(basename_train, basename_test, basename_pred):
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    print num_files
    for midx in range(num_files):
        classifier = get_trained_classifier(basename_train, midx)
        for pidx in [midx-1, midx, midx+1]:
            if( pidx >= 0 and pidx < num_files):    
                top_n_predicted = get_prediction(classifier, basename_test, pidx)
                dump_predicted_n(top_n_predicted, basename_pred, midx, pidx)

def get_direct_predicted(basename_train, basename_test, basename_pred):
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    print num_files
    for midx in range(num_files):
        train_filename = train_basepath +str(midx)+".csv.gz"
        classifier = get_trained_classifier(train_filename)
        test_filename = basepath +str(midx)+".csv.gz"
        top_n_predicted = get_prediction(classifier, test_filename)
        top_n_filename = basepath+str(midx)+"_"+str(midx)+".csv"
        dump_predicted_n(top_n_predicted, top_n_filename)
        print "Done", midx

def get_xy_predicted(basename_train, basename_test, basename_pred, x_rng, y_rng):
    num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1
    print num_x, num_y
    for idx_x in range(num_x):
        for idx_y in range(num_y):
            print idx_x, idx_y
            train_filename = basename_train+str(idx_x)+"_"+str(idx_y)+".csv"
            classifier = get_trained_classifier(train_filename)
            test_filename = basename_test+str(idx_x)+"_"+str(idx_y)+".csv"
            top_n_predicted = get_prediction(classifier, test_filename)
            top_n_filename = basename_pred+str(idx_x)+"_"+str(idx_y)+".csv"
            dump_predicted_n(top_n_predicted, top_n_filename)
    # for midx in range(num_files):
    #     classifier = get_trained_classifier(basename_train, midx)  
    #     top_n_predicted = get_prediction(classifier, basename_test, midx)
    #     dump_predicted_n(top_n_predicted, basename_pred, midx, midx)
    #     print "Done", midx


# Based on Standard dev of y value we find
MAX_MODEL_PER_TEST_FILE = 3
def get_predicted_idx_value(basepath, tidx, total_entries, num_files):
    predicted_value = np.zeros([total_entries, MAX_MODEL_PER_TEST_FILE*3], dtype='f4')
    predicted_idx = np.zeros([total_entries, MAX_MODEL_PER_TEST_FILE*3], dtype='i4')
    #print num_files
    for midx in [tidx-1, tidx, tidx+1]:
        if(midx >=0 and midx < num_files):
            infilename = basepath+str(midx)+"_"+str(tidx)+".csv"
            aridx = midx-tidx+1
            with open(infilename) as infile:
                num_entires = 0
                for line in infile:
                    record = [float(x) for x in line.strip().split(",")]
                    predicted_value[num_entires][3*aridx+0] = record[1]
                    predicted_value[num_entires][3*aridx+1] = record[3]
                    predicted_value[num_entires][3*aridx+2] = record[5]
                    predicted_idx[num_entires][3*aridx+0] = int(record[0])
                    predicted_idx[num_entires][3*aridx+1] = int(record[2])
                    predicted_idx[num_entires][3*aridx+2] = int(record[4])
                    num_entires = num_entires + 1
    return (predicted_value, predicted_idx)

def get_best_n_idx(predicted_value, predicted_idx, id_to_idx):
    n = 3
    top_n = np.argsort(predicted_value)[:,:-n-1:-1]

    row_id = np.arange(len(predicted_value))
    col0 = predicted_idx[row_id, top_n[:,0]]
    col1 = predicted_idx[row_id, top_n[:,1]]
    col2 = predicted_idx[row_id, top_n[:,2]]

    fileid = (top_n/3).astype('i4')
    #print fileid.shape, col0.shape

    pred0 = id_to_idx[col0]
    pred1 = id_to_idx[col1]
    pred2 = id_to_idx[col2]
    #print top_n.shape, predict.shape, col0.shape, col1.shape
    top_predicted = np.vstack((pred0, pred1, pred2, col0, col1, col2)).transpose()
    return top_predicted

def dump_pred(top_predicted, basefilename, testfileidx):
    top_n_filename = basefilename+str(testfileidx)+".csv"
    np.savetxt(top_n_filename, top_predicted, delimiter=",", fmt="%d,%d,%d, %d, %d, %d");

def get_id_to_idx(filename):
    with open(filename) as infile:
        id_to_idx = []
        for line in islice(infile,1,None):
            record = int(float(line.strip().split(",")[0]))
            id_to_idx.append(record)
    return np.array(id_to_idx)


def merge_predicted(pred_basefile, tn_pred_basefile):
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    #fileid_localid_to_labelid = get_fileid_localid_to_labelid()
    print id_to_idx.shape
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    for testfileidx in range(num_files):
        start = time.time()
        filename = pred_basefile+str(testfileidx)+"_"+str(testfileidx)+".csv"
        total_entries = ut.get_num_records(filename, 1)
        predicted_value, predicted_idx = get_predicted_idx_value(pred_basefile, testfileidx, total_entries, num_files)
        top_predicted = get_best_n_idx(predicted_value, predicted_idx, id_to_idx)
        # #print top_predicted[total_entries-10], predicted_idx[total_entries-10], predicted_value[total_entries-10]
        dump_pred(top_predicted, tn_pred_basefile, testfileidx)
        print testfileidx, total_entries, predicted_value.shape, predicted_idx.shape, time.time() - start

def merge_file(tn_pred_basefile, tnm_pred_file):
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    with open(tnm_pred_file, "w") as outfile:
        outfile.write("row_id,place_id\n");
        row_id = 0
        for i in range(num_files):
            infilename = tn_pred_basefile+str(i)+".csv"
            with open(infilename) as infile:
                for line in infile:
                    split_record = line.strip().split(",")
                    join_record = " ".join(split_record)
                    outfile.write(str(row_id)+","+join_record+"\n");
                    row_id = row_id + 1
    ut.zip_and_delete([tnm_pred_file])


def get_record_id_to_file_id(filename):
    record_id_to_file_id = []
    with open(filename) as infile:
        for line in infile:
            fileid = [int(float(x)) for x in line.strip().split(",")]
            record_id_to_file_id.append(fileid)
    return np.array(record_id_to_file_id)

def get_fileid_array(filename_base, numfiles):
    fileid_array = []
    for idx in range(numfiles):
        filename = filename_base+str(idx)+"_"+str(idx)+".csv"
        fileid = open(filename)
        fileid_array.append(fileid)
    return fileid_array

def close_fileid_array(fileid_array):
    for fileid in fileid_array:
        fileid.close()

def create_tnm(pred_basefile, record_id_to_file_id_file, tnm_pred_file):
    numfiles = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    record_id_to_file_id = get_record_id_to_file_id(record_id_to_file_id_file)
    pred_fileid = get_fileid_array(pred_basefile, numfiles)
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    with open(tnm_pred_file, "w") as tnmoutfile:
        tnmoutfile.write("row_id,place_id\n");
        for row_id in range(len(record_id_to_file_id)):
            infileid = pred_fileid[record_id_to_file_id[row_id]]
            line = infileid.readline();
            record = [int(float(x)) for x in line.strip().split(",")]
            pred0 = id_to_idx[record[0]]; pred1 = id_to_idx[record[2]]; pred2 = id_to_idx[record[4]]
            join_record = str(pred0)+" "+str(pred1)+" "+str(pred2)
            tnmoutfile.write(str(row_id)+","+join_record+"\n");
    close_fileid_array(pred_fileid)
    ut.zip_and_delete([tnm_pred_file])
    return

def create_xy_tnm(pred_basefile, record_id_to_file_id_file, tnm_pred_file, x_rng, y_rng):
    #numfiles = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    num_x = ((ct.MAX_XY_VALUE-1)/x_rng)+1; num_y = ((ct.MAX_XY_VALUE-1)/y_rng)+1
    record_id_to_file_id = get_record_id_to_file_id(record_id_to_file_id_file)
    pred_fileid = ut.get_xy_fileid_array(pred_basefile, num_x, num_y, False)
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    with open(tnm_pred_file, "w") as tnmoutfile:
        tnmoutfile.write("row_id,place_id\n");
        for row_id in range(len(record_id_to_file_id)):
            infileid = pred_fileid[record_id_to_file_id[row_id][0]][record_id_to_file_id[row_id][1]]
            line = infileid.readline();
            record = [int(float(x)) for x in line.strip().split(",")]
            pred0 = id_to_idx[record[0]]; pred1 = id_to_idx[record[2]]; pred2 = id_to_idx[record[4]]
            join_record = str(pred0)+" "+str(pred1)+" "+str(pred2)
            tnmoutfile.write(str(row_id)+","+join_record+"\n");
    ut.close_xy_fileid_array(pred_fileid, num_x, num_y)
    ut.zip_and_delete([tnm_pred_file])
    return

def create_tn(pred_basefile, tn_pred_basefile):
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    for testfileidx in range(num_files):
        infilename = pred_basefile+str(testfileidx)+"_"+str(testfileidx)+".csv"
        outfilename = tn_pred_basefile+str(testfileidx)+".csv"
        with open(infilename) as infile, open(outfilename, "w") as outfile:
            for line in infile:
                record = [int(float(x)) for x in line.strip().split(",")]
                pred0 = id_to_idx[record[0]]; pred1 = id_to_idx[record[2]]; pred2 = id_to_idx[record[4]]
                outfile.write(str(pred0)+","+str(pred1)+","+str(pred2)+"\n")

def calculate_performance(test_basefile, tn_pred_basefile):
    num_files = ((ct.MAX_Y_VALUE-1)/ct.Y_STEP_SIZE)+1
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    num_sample = 0
    total_score = 0
    mean_score = 0
    for i in range(num_files):
        testfilename = test_basefile+str(i)+".csv.gz"
        tnpredfilename = tn_pred_basefile+str(i)+".csv"
        with open(tnpredfilename) as predfile, gzip.open(testfilename) as testfile:
            for line in islice(testfile, 1, None):
                pred = [int(float(x)) for x in predfile.readline().strip().split(",")]
                test = int(id_to_idx[int(float((line.strip().split(","))[5]))])
                #print test, pred
                if pred[0] == test:
                    total_score = total_score + 1
                    mean_score = mean_score + 1.0
                if pred[1] == test:
                    total_score = total_score + (1/2.0)
                if pred[2] == test:
                    total_score = total_score + (1/3.0)
                num_sample = num_sample + 1
        print i, num_sample, total_score, mean_score
    print num_sample, total_score/num_sample, mean_score/num_sample

def calculate_performance_single(test_file_name, start, end, tnm_pred_file_name):
    print test_file_name, tnm_pred_file_name
    num_sample = 0; total_score = 0; mean_score = 0;
    id_to_idx = get_id_to_idx(ct.UNIQUE_ID_PATH)
    with open(test_file_name) as testfile, gzip.open(tnm_pred_file_name) as predfile:
        predfile.readline()
        for line in islice(testfile, start, end):
            pred_option = predfile.readline().strip().split(",")
            pred = [int(float(x)) for x in pred_option[1].split(" ")]
            test = int(id_to_idx[int(float((line.strip().split(","))[5]))])
            #print test, pred
            if pred[0] == test:
                total_score = total_score + 1
                mean_score = mean_score + 1.0
            if pred[1] == test:
                total_score = total_score + (1/2.0)
            if pred[2] == test:
                total_score = total_score + (1/3.0)
            num_sample = num_sample + 1
    print num_sample, total_score/num_sample, mean_score/num_sample


if __name__ == "__main__":

    # get_predicted_n(ct.Y_BASED_SPLIT_TRAIN_PATH, None, ct.Y_BASED_SPLIT_EVAL_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_PATH)
    # merge_predicted(ct.Y_BASED_SPLIT_EVAL_PRED_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH)
    # merge_file(ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TNM_PATH)
    # calculate_performance(ct.Y_BASED_SPLIT_EVAL_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH)

    # get_direct_predicted(ct.Y_BASED_SPLIT_TRAIN_PATH, ct.Y_BASED_SPLIT_EVAL_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_PATH)
    # print "Completed prediction"
    # #direct_pred_file(ct.Y_BASED_SPLIT_EVAL_PRED_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TNM_PATH)
    # #ut.zip_and_delete([ct.Y_BASED_SPLIT_EVAL_PRED_TNM_PATH])
    # #calculate_performance(ct.Y_BASED_SPLIT_EVAL_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH)
    # record_id_to_file_id_file = ct.Y_BASED_SPLIT_EVAL_PATH+ct.Y_BASED_SPLIT_REC_ID_TO_FIL_ID_NAME
    # create_tnm(ct.Y_BASED_SPLIT_EVAL_PRED_PATH, record_id_to_file_id_file, ct.Y_BASED_SPLIT_EVAL_PRED_TNM_PATH)
    # print "Completed mergedfile creation"
    # create_tn(ct.Y_BASED_SPLIT_EVAL_PRED_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH)
    # print "Completed non merged top n"
    # calculate_performance(ct.Y_BASED_SPLIT_EVAL_PATH, ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH)
    # print "completed performance"
    # total_records = ut.get_num_records(ct.TRAIN_ALL_INT_PATH, 1)
    # num_train_records = int(ct.TRAINING_PERCENT*total_records)
    # calculate_performance_single(ct.TRAIN_ALL_INT_PATH, num_train_records+1, total_records, ct.Y_BASED_SPLIT_EVAL_PRED_TNM_PATH+".gz")



    # get_predicted_n(ct.Y_BASED_SPLIT_TRAIN_PATH, None, ct.Y_BASED_SPLIT_TEST_PATH, ct.Y_BASED_SPLIT_TEST_PRED_PATH)
    # print "done with prediction"
    # merge_predicted(ct.Y_BASED_SPLIT_TEST_PRED_PATH, ct.Y_BASED_SPLIT_TEST_PRED_TN_PATH)
    # print "done with merge predicted"
    # merge_file(ct.Y_BASED_SPLIT_TEST_PRED_TN_PATH, ct.Y_BASED_SPLIT_TEST_PRED_TNM_PATH)
    # print "Done with merge file"

    # get_direct_predicted(ct.Y_BASED_SPLIT_FULL_TRAIN_PATH, ct.Y_BASED_SPLIT_TEST_PATH, ct.Y_BASED_SPLIT_TEST_PRED_PATH)
    # #get_direct_predicted(ct.Y_BASED_SPLIT_TRAIN_PATH, ct.Y_BASED_SPLIT_TEST_PATH, ct.Y_BASED_SPLIT_TEST_PRED_PATH)
    # print "Completed prediction"
    # record_id_to_file_id_file = ct.Y_BASED_SPLIT_TEST_PATH+ct.Y_BASED_SPLIT_REC_ID_TO_FIL_ID_NAME
    # create_tnm(ct.Y_BASED_SPLIT_TEST_PRED_PATH, record_id_to_file_id_file, ct.Y_BASED_SPLIT_TEST_PRED_TNM_PATH)
    # print "Completed mergedfile creation"


    # get_xy_predicted(ct.XY_SD_SP_TR80_2SD_PATH, ct.XY_SD_SP_EVAL_2SD_PATH, ct.XY_SD_SP_EVAL_PRED_2SD_PATH, ct.X_RNG, ct.Y_RNG)
    # record_id_to_file_id_file = ct.XY_SD_SP_EVAL_2SD_PATH+ct.Y_BASED_SPLIT_REC_ID_TO_FIL_ID_NAME
    # create_xy_tnm(ct.XY_SD_SP_EVAL_PRED_2SD_PATH, record_id_to_file_id_file, ct.XY_SD_SP_EVAL_PRED_TNM_2SD_PATH, ct.X_RNG, ct.Y_RNG)
    # total_records = ut.get_num_records(ct.TRAIN_ALL_INT_PATH, 1)
    # num_train_records = int(ct.TRAINING_PERCENT*total_records)
    # calculate_performance_single(ct.TRAIN_ALL_INT_PATH, num_train_records, total_records, ct.XY_SD_SP_EVAL_PRED_TNM_2SD_PATH+".gz")


    # get_xy_predicted(ct.XY_SD_SP_TR_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PATH, ct.XY_SD_SP_EVAL_TS_2SD_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    # record_id_to_file_id_file = ct.XY_SD_SP_EVAL_TS_2SD_PATH+ct.REC_ID_TO_FIL_ID_NAME
    # create_xy_tnm(ct.XY_SD_SP_EVAL_TS_2SD_PRED_PATH, record_id_to_file_id_file, ct.XY_SD_SP_EVAL_TS_2SD_PRED_TNM_PATH, ct.X_RNG, ct.Y_RNG)
    # calculate_performance_single(ct.SPLIT_EVAL_ALL_INT_PATH, 1, ut.get_num_records(ct.SPLIT_EVAL_ALL_INT_PATH, 1), ct.XY_SD_SP_EVAL_TS_2SD_PRED_TNM_PATH+".gz")


    get_xy_predicted(ct.XY_SD_SP_FUTR_PATH, ct.XY_SD_SP_TEST_PATH, ct.XY_SD_SP_TEST_PRED_PATH, ct.X_RNG, ct.Y_RNG)
    record_id_to_file_id_file = ct.XY_SD_SP_TEST_PATH+ct.REC_ID_TO_FIL_ID_NAME
    create_xy_tnm(ct.XY_SD_SP_TEST_PRED_PATH, record_id_to_file_id_file, ct.XY_SD_SP_TEST_PRED_TNM_PATH, ct.X_RNG, ct.Y_RNG)



