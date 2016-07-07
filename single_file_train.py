
from itertools import islice
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, datasets
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



from sklearn import preprocessing
import time
import constants as ct
import gzip

# Feature Engineering Lessons
# Lesson 1:
# Do not use features like numYear, numDays which do not have any periodicity in Time features
# this is because if the sample are on a time line the real values would lie in different part of the time line
# eg. if num days was used as features, then train samples would have values less than the max time value in train set
# This would give good results with cross validation (eval set) since eval set is part of train set and would have values in the time
# However in case of test values the num days would lie outside the values used for train and so would not produce good results
# Lesson 2:
# To bring in the wrap around feature say across 24 hours or across a month i.e. 23 Hours and 1 are close enough values while a 
# actual difference would cause them to be shown as separate values. To avoid this one suggestion was to multiply the values by a sine 
# so that it forms a periodic function. But when done so it maps 0-6-12-18-24 to 0-(1)-0-(-1)-0. Thus it maps 0-6 to 0-1 and 6-12 to 1-0 
# So does not differentiate between values from 6-12-18 is same as values between 18-24-6 and so does not capture day night difference properly
# Lesson 3:
# When creating eval data from train data, Make the eval data as close as possible to the test data. In this case, the test data had the time field 
# after the max time value available in the train data. Lets the max value in train was 786239. The test data started from 786242 and ended in 1006589
# So while preparing eval data move all the record in the time range (786242-220147.76) to 786242 as eval data (assuming 28% and data is evenly distributed in time)

def get_train_data(filename):
    NUM_FEATURES = 6
    with open(filename) as infile:
        num_entires = 0
        for line in islice(infile,1,None):
                num_entires = num_entires + 1
        print num_entires
        feature = np.empty([num_entires, NUM_FEATURES], dtype='i4')
        #feature = np.empty([num_entires, NUM_FEATURES])

        label = np.empty([num_entires, 1], dtype='i4').ravel()
        row_id = 0
        infile.seek(0)
        for line in islice(infile,1,None):
                data_element = line.strip().split(",")
                data_element = [int(float(x)) for x in data_element]
                feature[row_id][0] = data_element[1]
                feature[row_id][1] = data_element[2]
                feature[row_id][2] = np.log2(data_element[3])*100

                # Scale accuracy to evenly distribute the values
                #feature[row_id][2] = ((data_element[3])*10000)/115

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
                # feature[row_id][6] = int(DiFW/7) - Seems to decrease
                feature[row_id][5] = DiM
                # feature[row_id][7] = DiY - Seems to decrease
                # feature[row_id][7] = Minute                
                # feature[row_id][7] = HiD12

                # if(NUM_FEATURES >= 4):
                #     T = data_element[4]/(24*60.0)
                #     #T = data_element[4]*786239.0/(24*60.0)
                #     D = int(T)
                #     H = int(24*(T-D))

                #     feature[row_id][3] = H
                #     #feature[row_id][3] = int(24*((np.sin(2*np.pi*data_element[4]/(24*60.0))+1)/2))
                #     # print data_element[4], H, feature[row_id][3], 24*((np.sin(2*np.pi*data_element[4]/(24*60.0))+1)/2), row_id, 24*((np.sin(2*np.pi*row_id/(24*60.0))+1)/2)
                # if(NUM_FEATURES >= 5):
                #     feature[row_id][4] = (D%7)
                #     feature[row_id][5] = (D%30)
                #     feature[row_id][6] = 60*(24*(T - D) - H)
                #     feature[row_id][7] = (D%365)
                #     feature[row_id][8] = (D%120)

                label[row_id] = data_element[5]
                #print feature[row_id], label[row_id]
                row_id = row_id + 1
        print feature.shape, label.shape
        #print feature, label
	#size = feature.shape[0]
	#ds_feature = feature[0:size/2]
	#ds_label = label[0:size/2]
    return (feature, label)





def get_performance(np_feature, np_label, classifier):
    #print np_feature.shape, np_label.shape
    NUM_TRAIN = int(0.8*np_feature.shape[0])
    #print NUM_TRAIN

    # weights has two options 'uniform', 'distance'
    classifier.fit(np_feature[:NUM_TRAIN], np_label[:NUM_TRAIN,])
    #classifier.fit(np_feature, np_label)
    #print "Classifier done"
    predict = classifier.predict(np_feature[NUM_TRAIN:])
    compare = predict==np_label[NUM_TRAIN:]
    return compare.mean()

def get_performance(tr_feature, tr_label, ev_feature, ev_label,classifier):
    # weights has two options 'uniform', 'distance'
    classifier.fit(tr_feature, tr_label)
    #classifier.fit(np_feature, np_label)
    #print "Classifier done"
    predict = classifier.predict(ev_feature)
    compare = predict==ev_label
    return compare.mean()


def get_pred_file(fulltestpredname):
    with open(fulltestpredname) as infile:
        num_entires = 0
        for line in infile:
            num_entires = num_entires + 1
        print num_entires
        pred_idx = np.zeros((num_entires, 3))
        pred_val = np.zeros((num_entires, 3))
        line_idx = 0
        infile.seek(0)
        for line in infile:
            record = line.strip().split(",")
            pred_idx[line_idx][0] = int(float(record[0]))
            pred_idx[line_idx][1] = int(float(record[2]))
            pred_idx[line_idx][2] = int(float(record[4]))
            pred_val[line_idx][0] = (float(record[1]))
            pred_val[line_idx][1] = (float(record[3]))
            pred_val[line_idx][2] = (float(record[5]))
            line_idx = line_idx + 1
    return (pred_idx, pred_val)

def get_pred_tn_file(fulltestpred_tn_name):
    with open(fulltestpred_tn_name) as infile:
        num_entires = 0
        for line in infile:
            num_entires = num_entires + 1
        print num_entires
        pred_idx = np.zeros((num_entires, 3))
        line_idx = 0
        infile.seek(0)
        for line in infile:
            record = line.strip().split(",")
            pred_idx[line_idx][0] = int(float(record[3]))
            pred_idx[line_idx][1] = int(float(record[4]))
            pred_idx[line_idx][2] = int(float(record[5]))
            line_idx = line_idx + 1
    return (pred_idx)

def compare_outputs(tr_feature, tr_label, ev_feature, ev_label, pred_idx, pred_tn_idx, classifier):

    # weights has two options 'uniform', 'distance'
    classifier.fit(tr_feature, tr_label)
    #classifier.fit(np_feature, np_label)
    #print "Classifier done"
    predict = classifier.predict(ev_feature)
    # predict_prob = classifier.predict_proba(ev_feature)
    # top_0 = np.argmax(predict_prob, axis=1)

    # classes = classifier.classes_ 
    # row_id = np.arange(len(predict))

    # top_class_0 = classes[top_0]
    # col_0 = predict_prob[row_id, top_0]
    # predict_prob[row_id, top_0] = -1.0
    # top_1 = np.argmax(predict_prob, axis=1)
    # top_class_1 = classes[top_1]
    # col_1 = predict_prob[row_id, top_1]
    # predict_prob[row_id, top_1] = -1.0
    # top_2 = np.argmax(predict_prob, axis=1)
    # top_class_2 = classes[top_2]
    # col_2 = predict_prob[row_id, top_2]
    print pred_idx.shape, pred_tn_idx.shape

    compare = predict==ev_label
    compare_ft = pred_idx[:,0]==ev_label
    compare_ft_tn = pred_tn_idx[:,0]==ev_label
    #compare_pred_prob = top_class_0==ev_label
    print compare_ft_tn.mean(), compare_ft.mean(), compare.mean()
    return compare.mean()


# - Cannot combin the result of two separate trees trained on different data sets. 
#     - This is because they are not normalised across all the data sets
#     - A pred value could have more close estimates in one data set while could have 
# print ev_label[i], pred_idx_0[i], pred_idx_1[i], pred_val_0[i], pred_val_1[i], pred_tn_idx[i]
# 9443 [  9443.  17716.  38314.] [ 41462.  95578.  22427.] [ 0.36  0.21  0.06] [ 0.5   0.2   0.12] [ 41462.   9443.  17716.]
# 58883 [ 58883.  65165.  98355.] [ 107367.   57694.   89212.] [ 0.49  0.3   0.04] [ 0.8  0.1  0.1] [ 107367.   58883.   65165.]
# 67122 [ 67122.  24940.  36350.] [ 19581.   7775.     54.] [ 0.52  0.11  0.1 ] [ 0.9  0.1  0. ] [ 19581.  67122.  24940.]
# 16298 [ 16298.  53611.  59317.] [ 46795.  27902.  47713.] [ 0.4   0.13  0.11] [ 0.97  0.01  0.01] [ 46795.  16298.  53611.]
# 75094 [ 75094.  78542.  44221.] [ 21704.  80718.   4590.] [ 0.42  0.26  0.1 ] [ 0.44  0.19  0.1 ] [ 21704.  75094.  78542.]
# 64522 [ 64522.  34620.  56005.] [ 58428.  96204.  27702.] [ 0.31  0.13  0.1 ] [ 0.5  0.3  0.1] [ 58428.  64522.  96204.]
# 59905 [ 59905.  22288.  33630.] [  3298.  68189.  60259.] [ 0.37  0.26  0.11] [ 0.82  0.02  0.02] [  3298.  59905.  22288.]
# 37697 [ 37697.  53723.  81624.] [ 74958.  40095.  21131.] [ 0.49  0.13  0.12] [ 0.5   0.29  0.11] [ 74958.  37697.  40095.]

def compare_single_cross(ev_label, pred_idx_0, pred_idx_1, pred_val_0, pred_val_1, pred_tn_idx):
    total_entries = 0; num_wrong_predictions = 0; num_improved_prediction = 0;
    for i in range(ev_label.shape[0]):
        if( ev_label[i] == pred_idx_0[i][0] and ev_label[i] != pred_tn_idx[i][0] ):
            print ev_label[i], pred_idx_0[i], pred_idx_1[i], pred_val_0[i], pred_val_1[i], pred_tn_idx[i]
            num_wrong_predictions = num_wrong_predictions + 1
        # if( ev_label[i] != pred_idx[i][0] and ev_label[i] == pred_tn_idx[i][0] ):
        #     # print ev_label[i], pred_idx[i], pred_tn_idx[i]
        #     num_improved_prediction = num_improved_prediction + 1
        total_entries = total_entries + 1
    compare_ft = pred_idx[:,0]==ev_label
    compare_ft_tn = pred_tn_idx[:,0]==ev_label
    print total_entries, num_wrong_predictions, num_improved_prediction, compare_ft.mean(), compare_ft_tn.mean()



DR_FILENAME = "../../data/fb/distributed_record/dr2.csv"

#Weird distribution of data:
#(1035578, 4) (1035578,) - Actual distribution without downsampling
#(1035578, 4) (1035578,)
#6499
#- Downsampling by skipping intermediate samples a:b:N does not provide a corresponding downsampling of the number of train examples
#(1035578, 4) (1035578,)
#(103558, 4) (103558,)
#5370
#- Downsampling by reducing the size of the samples a:b also does not reduce the number of uniques samples
#(1035578, 4) (1035578,)
#(103558, 4) (103558,)
#5370

def calculate_distance(distances):
    return distances ** -2

#trainfile = ct.Y_BASED_SPLIT_TRAIN_PATH+"0.csv.gz"
#trainfile = ct.XY_SD_SP_TR80_3SD_PATH+"5_39.csv.gz"
#trainfile = ct.XY_SD_SP_TR80_2SD_PATH+"7_39.csv"
#trainfile = ct.XY_SD_SP_TR80_PATH+"7_39.csv"
trainfile = ct.XY_SD_SP_TR_TS_2SD_PATH+"5_39.csv"
(tr_feature_0, tr_label_0) = get_train_data(trainfile)

# trainfile = ct.Y_BASED_SPLIT_TRAIN_PATH+"1.csv.gz"
# (tr_feature_1, tr_label_1) = get_train_data(trainfile)
#evalfile = ct.Y_BASED_SPLIT_EVAL_PATH+"0.csv.gz"
#evalfile = ct.XY_SD_SP_EVAL_3SD_PATH+"5_39.csv.gz"
#evalfile = ct.XY_SD_SP_EVAL_2SD_PATH+"7_39.csv"
#evalfile = ct.XY_SD_SP_EVAL_PATH+"7_39.csv"
evalfile = ct.XY_SD_SP_EVAL_TS_2SD_PATH+"5_39.csv"
(ev_feature, ev_label) = get_train_data(evalfile)

# print tr_feature.shape, np_label_without_pre_processing.shape
# le = preprocessing.LabelEncoder()
# np_label = le.fit_transform(np_label_without_pre_processing)
# print len(le.classes_)
# NUM_TRAIN = int(0.8*np_feature.shape[0])
# print NUM_TRAIN

start_time = time.time()    
#classifier = AdaBoostClassifier()
classifier = RandomForestClassifier(n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1)
#classifier = GaussianNB()
#classifier = KNeighborsClassifier(n_neighbors=100, weights=calculate_distance, metric='manhattan', n_jobs = -1)

mean = get_performance(tr_feature_0, tr_label_0, ev_feature, ev_label, classifier)
#print np_feature, np_label
total_time = time.time() - start_time
#print n_estimators, mean, total_time
print "RandomForestClassifier", mean, total_time

# n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1
# 0.565025174997 77.0155968666
# n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1
# 0.571902247329 112.773393869
# n_estimators=20, max_features=4, class_weight="balanced", max_depth=18, n_jobs = -1
# 0.566498833354 88.7793989182
# (n_estimators=20, max_features=4, class_weight="balanced", max_depth=20, n_jobs = -1
# 0.576446027263 111.526978016

# 3sd, 
# (5_39) RandomForestClassifier 0.579884563429 52.3995029926
# (6_39) RandomForestClassifier 0.517583195657 50.4737260342
# (7_39) RandomForestClassifier 0.502363693665 44.3755319118

# 2sd,
# (5_39) RandomForestClassifier 0.588235294118 23.7585639954
# (6_39) RandomForestClassifier 0.527377861695 23.9662160873
# (7_39) RandomForestClassifier 0.508771929825 21.8937139511

# 1sd
# (5_39) 0.533278262922 7.62824606895
# (6_39) 0.579761758566 8.07797288895
# (7_39) 0.506986027944 7.8070640564








# n_estimators 5 0.44775300572 6.64044094086
# n_estimators 10 0.474320065367 13.4336340427
# n_estimators 15 0.481592155947 15.8454101086
# n_estimators 20 0.485841017859 21.0308189392
# n_estimators 25 0.487673631376 30.162968874

# max_depth 15 0.469160733045 19.4723770618
# max_depth 16 0.475977588421 24.986369133
# max_depth 17 0.482806116494 24.3526339531
# max_depth 18 0.487556904401 29.0647919178
# max_depth 19 0.490626823859 68.5050020218
# Killed

# for max_depth in [15, 16, 17, 18, 19, 20, 21]:
#     start_time = time.time()    
#     #classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, n_jobs = -1)
#     classifier = ExtraTreesClassifier(n_estimators=25, max_features=4, class_weight="balanced", max_depth=max_depth, n_jobs = -1)
#     mean = get_performance(tr_feature_0, tr_label_0, ev_feature, ev_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "max_depth", max_depth, mean, total_time


# n_neighbors 5 0.41593323217 7.71996593475
# n_neighbors 10 0.439488735847 11.626611948
# n_neighbors 15 0.445196684954 16.0978829861
# n_neighbors 20 0.447402824793 18.2453620434
# n_neighbors 25 0.449037002451 20.9475090504
# n_neighbors 50 0.448009805066 31.5324239731
# n_neighbors 100 0.443714252364 59.8706729412
# n_neighbors 200 0.433547332789 103.002221107
# n_neighbors 500 0.403933699078 263.634507895

# for n_neighbors in [50,100,200,500,1000]:
#     start_time = time.time()    
#     #classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, n_jobs = -1)
#     classifier = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights='uniform')
#     mean = get_performance(tr_feature_0, tr_label_0, ev_feature, ev_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "n_neighbors", n_neighbors, mean, total_time


# There is a performance improvement of 1% but increases the resource usage line time to 186ms and the memory shoots upto 65Gb
# Multiple 0.496241391386 185.483706951
# Direct 0.486961596825 52.5012278557
# start_time = time.time()
# tr_feature = np.concatenate((tr_feature_0, tr_feature_1), axis=0)
# tr_label = np.concatenate((tr_label_0, tr_label_1), axis=0)
# classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, n_jobs = -1)
# mean = get_performance(tr_feature, tr_label, ev_feature, ev_label, classifier)
# #print np_feature, np_label
# total_time = time.time() - start_time
# #print n_estimators, mean, total_time
# print "Multiple", mean, total_time

# start_time = time.time()    
# classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, n_jobs = -1)
# mean = get_performance(tr_feature_0, tr_label_0, ev_feature, ev_label, classifier)
# #print np_feature, np_label
# total_time = time.time() - start_time
# #print n_estimators, mean, total_time
# print "Direct", mean, total_time




# fulltestfile = ct.Y_BASED_SPLIT_EVAL_PRED_PATH+"0_0.csv"
# (pred_idx_0, pred_val_0) = get_pred_file(fulltestfile)
# fulltestfile = ct.Y_BASED_SPLIT_EVAL_PRED_PATH+"1_0.csv"
# (pred_idx_1, pred_val_1) = get_pred_file(fulltestfile)
# fulltest_tn_file = ct.Y_BASED_SPLIT_EVAL_PRED_TN_PATH+"0.csv"
# pred_tn_idx = get_pred_tn_file(fulltest_tn_file)
#compare_single_cross(ev_label, pred_idx_0, pred_idx_1, pred_val_0, pred_val_1, pred_tn_idx)

# start_time = time.time()    
# classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, n_jobs = 16)
# #mean = get_performance_1(tr_feature, tr_label, ev_feature, ev_label, classifier)
# mean = compare_outputs(tr_feature, tr_label, ev_feature, ev_label, pred_idx, pred_tn_idx, classifier)
# print "classifier details", classifier.n_classes_, classifier.n_features_, classifier.feature_importances_ 
# # classifier.oob_score_, classifier.oob_decision_function_

# #print np_feature, np_label
# total_time = time.time() - start_time
# #print n_estimators, mean, total_time
# print "n_estimators", 10, "max_depth ", 18, mean, total_time


# DS = 1
# fig = plt.figure()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0,4500,0,20))
# #plt.scatter(np_feature[:NUM_TRAIN:DS, 1], np_feature[:NUM_TRAIN:DS, 0], c=np_label[:NUM_TRAIN:DS], s=np_label[:NUM_TRAIN:DS], cmap='Accent', alpha=0.2)
# plt.scatter(np_feature[:NUM_TRAIN:DS, 1], np_label[:NUM_TRAIN:DS])#,  c=np_label[:NUM_TRAIN:DS], cmap='Accent', alpha=0.2)

# fig.savefig("La_Y_Train_2.jpg")
# fig = plt.figure()
# plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
# plt.axis((0,4500,0,20))
# #plt.scatter(np_feature[NUM_TRAIN::DS, 1], np_feature[NUM_TRAIN::DS, 0], c=np_label[NUM_TRAIN::DS], s=np_label[:NUM_TRAIN:DS], cmap='Accent', alpha=0.2)
# plt.scatter(np_feature[NUM_TRAIN::DS, 1], np_label[NUM_TRAIN::DS])#,  c=np_label[:NUM_TRAIN:DS], cmap='Accent', alpha=0.2)

# fig.savefig("La_Y_Test_2.jpg")

# max_features=4
# 1 0.408517257572 30.6527481079
# 3 0.447845738436 23.8121051788
# 5 0.476711082414 23.6908380985
# 10 0.499970650387 33.0557940006
# 15 0.508159192299 50.8873510361
# 20 0.512488260155 75.6605100632
# 25 Killed
# max_feature=4, criterion="entropy"
# 1 0.409926038976 96.5561568737
# 3 0.447816388824 71.2837760448
# 5 0.477503521954 69.1016261578
# 10 0.49909016201 94.0479850769
# class_weight="balanced", max_feature=4
# 1 0.398714486969 26.8404128551
# 3 0.445776590749 25.7284610271
# 5 0.475170227753 27.8478109837
# 10 0.498958088753 32.3348090649
# 15 0.508379314393 52.4125208855
# 20 0.51215073961 76.3317677975

# max_depth variation values greater than 20  was not giving great benefit in the mean performance
# n_estimators=10, max_features=4, class_weight="balanced", max_depth=max_depth, n_jobs = 16
# 10 0.363040032872 12.5620849133
# 12 0.437177154262 15.7986459732
# 14 0.477151326602 20.0506291389
# 16 0.494540972059 25.5457699299
# 18 0.503199107772 29.099822998
# 20 0.499823902324 28.9197359085

# Finding the min between n_estimators and max_depth
# n_estimators  5 max_depth  10 0.344388354074 9.12294697762
# n_estimators  5 max_depth  12 0.429106010801 10.777203083
# n_estimators  5 max_depth  14 0.45935078657 13.7398359776
# n_estimators  5 max_depth  16 0.476021366518 16.6724948883
# n_estimators  5 max_depth  18 0.477166001409 19.312278986
# n_estimators  5 max_depth  20 0.480306409955 20.7513148785
# n_estimators 10 max_depth  10 0.361411129373 12.8218538761
# n_estimators 10 max_depth  12 0.436443413947 15.9735050201
# n_estimators 10 max_depth  14 0.477488847147 19.8346390724
# n_estimators 10 max_depth  16 0.495641582531 27.637198925 <- Looks like a good balabce between time, n_estimators and max_depth
# n_estimators 10 max_depth  18 0.501599553886 29.5023608208 <- Next best value
# n_estimators 10 max_depth  20 0.501805001174 30.2005839348
# n_estimators 15 max_depth  10 0.360266494482 18.2844548225
# n_estimators 15 max_depth  12 0.445160248885 22.790127039
# n_estimators 15 max_depth  14 0.48155376849 31.0101900101
# n_estimators 15 max_depth  16 0.50289093684 35.2244291306
# n_estimators 15 max_depth  18 0.511211552008 41.8954451084
# n_estimators 15 max_depth  20 0.509347851608 46.8513460159
# n_estimators 20 max_depth  10 0.373591218596 27.3507227898
# n_estimators 20 max_depth  12 0.444793378727 32.3116660118
# n_estimators 20 max_depth  14 0.482991899507 45.0472269058
# n_estimators 20 max_depth  16 0.506779760507 50.7809398174
# n_estimators 20 max_depth  18 0.513559521014 63.4972019196
# n_estimators 20 max_depth  20 0.515613993895 72.2631909847
# n_estimators 25 max_depth  10 0.369761094154 31.140335083
# n_estimators 25 max_depth  12 0.447728339986 40.2022299767
# n_estimators 25 max_depth  14 0.485046372388 47.5992000103
# n_estimators 25 max_depth  16 0.507689598497 58.8478329182
# n_estimators 25 max_depth  18 0.51555529467 69.4849069118
# n_estimators 25 max_depth  20 0.51907724818 76.3479449749
# Also looks like the memory is reduced with max_depth and we are able to run n_estimators 25 could be run without getting killed

# criterion="entropy" does not seem to improve performance but increases the time of execution so using default "gini"
#print np_feature, np_label
# for n_estimators in [5,10,15,20,25]:
#     for max_depth in [10, 12, 14, 16, 18, 20]:
#         start_time = time.time()    
#         classifier = RandomForestClassifier(n_estimators=n_estimators, max_features=4, class_weight="balanced", max_depth=max_depth, n_jobs = -1)
#         mean = get_performance(np_feature, np_label, classifier)
#         #print np_feature, np_label
#         total_time = time.time() - start_time
#         #print n_estimators, mean, total_time
#         print "n_estimators", n_estimators, "max_depth ", max_depth, mean, total_time

# min_samples_split 2 0.503874148861 40.0497980118
# min_samples_split 3 0.502333294201 43.9687180519
# min_samples_split 4 0.50181967598 45.9015538692
# min_samples_split 5 0.501511505048 39.9494240284
# min_samples_split 6 0.500234796901 38.0115339756
# min_samples_split 7 0.499691829068 38.2375929356
# min_samples_split 10 0.497431908899 37.1846170425
# min_samples_split 100 0.452204155905 24.6660830975
# min_samples_split 1000 0.254182319793 16.2944591045
# No big improvement with this parameter
# for min_samples_split in [2, 3, 4, 5, 6, 7, 10, 100, 1000]:
#     start_time = time.time()    
#     classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, min_samples_split=min_samples_split, n_jobs = -1)
#     mean = get_performance(tr_feature, tr_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "min_samples_split", min_samples_split, mean, total_time


# min_samples_leaf 1 0.502700164358 44.5286660194
# min_samples_leaf 10 0.486469828598 31.9440431595
# min_samples_leaf 100 0.404789856774 16.155632019
# min_samples_leaf 1000 0.173852430148 10.469326973
# for min_samples_leaf in [1, 10, 100, 1000]:
#     start_time = time.time()    
#     classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, min_samples_leaf=min_samples_leaf , n_jobs = -1)
#     mean = get_performance(tr_feature, tr_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "min_samples_leaf", min_samples_leaf, mean, total_time

# bootstrap True 0.502788213196 44.8854660988
# bootstrap False 0.421695233623 58.1373951435
# for bootstrap in [True, False]:
#     start_time = time.time()    
#     classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, bootstrap=bootstrap, n_jobs = -1)
#     mean = get_performance(tr_feature, tr_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "bootstrap", bootstrap, mean, total_time

# max_leaf_nodes 10 0.0120039915473 9.33673095703
# max_leaf_nodes 100 0.117765320498 13.9356729984
# max_leaf_nodes 1000 0.377920286452 21.9954619408
# max_leaf_nodes 1000 0.384788095797 21.6424229145
# max_leaf_nodes 2500 0.444881427565 27.529253006
# max_leaf_nodes 5000 0.469388354074 30.0643110275
# max_leaf_nodes 7500 0.47842803475 34.1547029018
# max_leaf_nodes 10000 0.482903850669 33.5769240856

# for max_leaf_nodes in [1000, 2500, 5000, 7500, 10000]:
#     start_time = time.time()    
#     classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, max_leaf_nodes=max_leaf_nodes , n_jobs = -1)
#     mean = get_performance(tr_feature, tr_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "max_leaf_nodes", max_leaf_nodes, mean, total_time

# oob_score True 0.502025123268 71.7044098377
# oob_score False 0.502010448462 41.3845670223
# for oob_score in [True, False]:
#     start_time = time.time()    
#     classifier = RandomForestClassifier(n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, oob_score=oob_score, n_jobs = -1)
#     mean = get_performance(tr_feature, tr_label, classifier)
#     #print np_feature, np_label
#     total_time = time.time() - start_time
#     #print n_estimators, mean, total_time
#     print "oob_score", oob_score, mean, total_time




# start_time = time.time()    
# classifier = LogisticRegression(n_jobs = 16)
# mean = get_performance(np_feature, np_label, classifier)
# #print np_feature, np_label
# total_time = time.time() - start_time
# print n_estimators, mean, total_time



# Calculating the performance metrics for the y split data across files
# print i, num_sample, total_score, mean_score
# 0 85670 50262.5 41665.0
# 1 176196 100779.5 83597.0
# 2 262531 148153.833333 122338.0
# 3 344950 196087.666667 162061.0
# 4 432187 245711.333334 203088.0
# 5 517889 293423.833333 242463.0
# 6 610163 344696.333333 284598.0
# 7 696014 391688.0 323533.0
# 8 783280 440524.499999 363902.0
# 9 874830 493009.833333 406894.0
# 10 967060 543502.499999 448465.0
# 11 1054792 592701.166666 489137.0
# 12 1142384 641766.833333 529516.0
# 13 1227301 689627.166667 569148.0
# 14 1313017 737225.500001 608268.0
# 15 1399674 785300.166668 648136.0
# 16 1488656 833802.500002 688170.0
# 17 1574119 881949.000002 727909.0
# 18 1664917 934292.833336 771402.0
# 19 1751390 982791.833336 811311.0
# 20 1838222 1031247.33334 851314.0
# 21 1924125 1080144.33334 891321.0
# 22 2005980 1126104.16667 929550.0
# 23 2093516 1175849.66667 970428.0
# 24 2181875 1226992.83333 1012812.0
# 25 2264769 1272789.33333 1050566.0
# 26 2355843 1325214.33333 1093956.0
# 27 2444235 1374553.0 1134453.0
# 28 2536504 1426498.66666 1177547.0
# 29 2626823 1476806.0 1219276.0
# 30 2716556 1526919.16666 1260605.0
# 31 2803288 1573485.66666 1298809.0
# 32 2889704 1620964.33333 1337626.0
# 33 2974936 1668320.66666 1376534.0
# 34 3062316 1718713.49999 1418531.0
# 35 3143921 1763829.99999 1456000.0
# 36 3233311 1814711.33332 1497898.0
# 37 3320422 1864714.66666 1539403.0
# 38 3406647 1912867.33332 1579094.0
# 39 3489175 1959517.66666 1617676.0
# 40 3573463 2007326.99999 1657027.0
# 41 3658473 2054265.49999 1695868.0
# 42 3743921 2102434.49999 1735607.0
# 43 3832553 2151351.16665 1775812.0
# 44 3924419 2201193.99999 1816344.0
# 45 4012073 2249984.49999 1856377.0
# 46 4097606 2297989.99999 1896080.0
# 47 4183658 2346809.33333 1936501.0
# 48 4275420 2397857.83333 1978582.0
# 49 4360017 2444003.33333 2016725.0
# 50 4447645 2493850.83333 2058070.0
# 51 4535560 2543620.5 2099383.0
# 52 4625081 2593591.33334 2140928.0
# 53 4713236 2642987.83334 2181642.0
# 54 4798439 2690339.83334 2220754.0
# 55 4886359 2740199.33334 2261941.0
# 56 4973465 2788060.50001 2301063.0
# 57 5062859 2837978.50001 2342206.0
# 58 5153199 2886661.16668 2381972.0
# 59 5243159 2937086.16668 2423377.0
# 60 5332555 2986600.33335 2464032.0
# 61 5420953 3034422.16668 2503493.0
# 62 5507961 3083292.00002 2543672.0
# 63 5591889 3130606.00002 2582781.0
# 64 5679134 3179949.00002 2623519.0
# 65 5767441 3230044.33336 2664931.0
# 66 5823604 3262380.83336 2691580.0
# print num_sample, total_score/num_sample, mean_score/num_sample
# 5823604 0.560199634686 0.462184585353



