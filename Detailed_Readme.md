
Begining:
About the data:
# print num_entries, len(unique_entries.keys()), min(unique_entries.values()), max(unique_entries.values()), sum(unique_entries.values())
# 29118022 108390 1 1849 29118021
# min, max, sum, sqrsum, mean, variance (std_deviation)
# x 0.0 100000.0 1.45583218023e+12 9.65655984118e+16 49997.633089 816588270.502 (28875.046)
# y 0.0 100000.0 1.45642737658e+12 9.71253765874e+16 50018.0739124 833768324.566 (28875.0467)
# a 1.0 1033.0 2412402560.0 5.83290633356e+11 82.8491221004 13167.9685903 (114.751769)
# t 1.0 786239.0 1.21425165572e+13 6.61969244232e+18 417010.350402 53442413019.3 (231176.151)

# Test
# 8607231
# x 0.0 100000.0 4.29622153113e+11 2.85161592468e+16 49914.0958472 821629850.32
# y 0.0 100000.0 4.30938061925e+11 2.87491404793e+16 50066.9799527 833412414.759
# a 1.0 1026.0 797477730.0 2.06853788674e+11 92.6520654552 15448.1539806
# t 786242.0 1006589.0 7.66442554218e+12 6.86066494628e+18 890463.558161 4156192244.82

# Data for x an y has only 4 decimal precision
# so Range is from 0 - 100000
# Accuracy range is 1033
# Time series - 786239

How is the evaluation done:
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

#Reading large files
#http://stackoverflow.com/questions/6475328/read-large-text-files-in-python-line-by-line-without-loading-it-in-to-memory

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

#http://stackoverflow.com/questions/28568034/getting-scikit-learn-randomforestclassifier-to-output-top-n-results

#  Mean Average Precision @3  : https://www.kaggle.com/wiki/MeanAveragePrecision - Still not fully clear


Results:
========

Wrong Results based on wrong sampling of data:
# Nearest neighbor algorithm: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.get_params
# weight, num_neighbors, algorithm, mean_correct_value, total_time
# uniform 1 kd_tree 0.749281473195 1.55174517632
# uniform 3 kd_tree 0.722209792617 3.0338640213
# uniform 5 kd_tree 0.704603734618 3.45425486565
# uniform 10 kd_tree 0.665330006024 4.44531583786
# uniform 15 kd_tree 0.636812666724 5.49731707573
# uniform 20 kd_tree 0.613905860081 6.59826517105
# uniform 25 kd_tree 0.595077876258 7.83416604996
# distance 1 kd_tree 0.749281473195 1.64854598045
# distance 3 kd_tree 0.746321314861 3.14151501656
# distance 5 kd_tree 0.739041390586 4.65441203117
# distance 10 kd_tree 0.717287668875 6.50194501877
# distance 15 kd_tree 0.700645383358 8.29478502274
# distance 20 kd_tree 0.686223216591 9.91831898689
# distance 25 kd_tree 0.673539282334 12.5876750946

# Random forest classifier
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
# num_estimators: number of trees in the forest
# num_estimators, mean_correct_value, total_time
# dr0
# 1 0.812735564926 8.86215400696
# 3 0.863264779279 27.1258850098
# 5 0.895516736942 47.5079510212
# 10 0.906376387574 106.926122904
# 15 0.91623784528 161.358649969
# 20 0.916685311075 240.075958014
# 25 0.916891833749 294.315912962
# dr1
# 1 0.844740057478 8.65076804161
# 3 0.872446608959 28.8303010464
# 5 0.888623104855 49.2628529072
# 10 0.903749849421 105.036150932
# 15 0.914436662135 157.265836954
# 20 0.91837753188 212.591512203
# 25 0.919358447056 290.361974955
# dr2
# 1 0.814358480176 9.26847600937
# 3 0.86720470815 29.4440858364
# 5 0.90002064978 51.8748180866
# 10 0.911894273128 111.555720091
# 15 0.916041437225 168.062342882
# 20 0.919448650881 235.267134905
# 25 0.92340652533 302.791250944
# Memeory seems to peak for num_estimator goes to 25. It seems to hit 20GB etc

Results from running various hyperparameters for Random Forest:
n_estimators=10, max_features=4, class_weight="balanced", max_depth=18, min_samples_split=min_samples_split, n_jobs = -1
#print "oob_score", oob_score, mean, total_time
# oob_score True 0.502025123268 71.7044098377
# oob_score False 0.502010448462 41.3845670223
#print n_estimators, mean, total_time
# max_leaf_nodes 10 0.0120039915473 9.33673095703
# max_leaf_nodes 100 0.117765320498 13.9356729984
# max_leaf_nodes 1000 0.377920286452 21.9954619408
# max_leaf_nodes 1000 0.384788095797 21.6424229145
# max_leaf_nodes 2500 0.444881427565 27.529253006
# max_leaf_nodes 5000 0.469388354074 30.0643110275
# max_leaf_nodes 7500 0.47842803475 34.1547029018
# max_leaf_nodes 10000 0.482903850669 33.5769240856
# bootstrap True 0.502788213196 44.8854660988
# bootstrap False 0.421695233623 58.1373951435
# min_samples_leaf 1 0.502700164358 44.5286660194
# min_samples_leaf 10 0.486469828598 31.9440431595
# min_samples_leaf 100 0.404789856774 16.155632019
# min_samples_leaf 1000 0.173852430148 10.469326973
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
# criterion="entropy" does not seem to improve performance but increases the time of execution so using default "gini"

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

One of the initial full run results with Random forest:
# Calculating the performance metrics for the y split data across files
# print i, num_sample, total_score, mean_score
# 0 85670 50262.5 41665.0
# 1 176196 100779.5 83597.0
# 2 262531 148153.833333 122338.0
# 3 344950 196087.666667 162061.0
# 4 432187 245711.333334 203088.0
# 5 517889 293423.833333 242463.0
# 6 610163 344696.333333 284598.0
..........
# print num_sample, total_score/num_sample, mean_score/num_sample
# 5823604 0.560199634686 0.462184585353


Have increased number of features from usual x,y, accuaracy to time based features like hour in a day, day in a week, day in a month, just day id, year etc
These features seem to give very good results on the eval set but when uploaded they provided a very bad result. there was a huge difference between evaluation results and actual results
This was because we did not split the eval data set based on time and have split it based on just file position, while the train data set specifically mentions this.
Most of the below results were based on this wrong eval data. 
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

To take care of the points across the boundary split thought of using 
        - a single classfier which combines three train splits (left, center and right) and trains the classifier
            - This caused the memory of Random forest to shoot up drastically So did not try
# There is a performance improvement of 1% but increases the resource usage line time to 186ms and the memory shoots upto 65Gb
# Multiple 0.496241391386 185.483706951 (Combining two train sets in one and having it as a single file) - to check for boundary cases acros split
# Direct 0.486961596825 52.5012278557

# n_neighbors 5 0.41593323217 7.71996593475
# n_neighbors 10 0.439488735847 11.626611948
# n_neighbors 15 0.445196684954 16.0978829861
# n_neighbors 20 0.447402824793 18.2453620434
# n_neighbors 25 0.449037002451 20.9475090504
# n_neighbors 50 0.448009805066 31.5324239731
# n_neighbors 100 0.443714252364 59.8706729412
# n_neighbors 200 0.433547332789 103.002221107
# n_neighbors 500 0.403933699078 263.634507895

a Run RF for left, center and right with the same eval/train set and merge the results based on the pred_prob values
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


Next calcualted the mean and std. deviation for every place_id and then used the mean value to choose whether the particular candidate was part of the split file or not
# Average 1-sd (std deviation) implies 70% of values would lie between +/- 1sd
# 108391 6750.0715741 206.229965204 106.961287001
# Average 2-sd imples 95% values lie between +/- 2sd
# 108391 13500.1431482 412.459930408 213.922574002
# Average 3-sd (std deviation) implies 99% of values would lie between +/- 3sd
# 108391 20250.2147223 618.689895612 320.883861004

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

- Feature Engineering (Time based Features):
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
        Later used two features sin and cos separately to encode this difference for each periodic feature
        # Lesson 3:
        # When creating eval data from train data, Make the eval data as close as possible to the test data. In this case, the test data had the time field 
        # after the max time value available in the train data. Lets the max value in train was 786239. The test data started from 786242 and ended in 1006589
        # So while preparing eval data move all the record in the time range (786242-220147.76) to 786242 as eval data (assuming 28% and data is evenly distributed in time)

MAX_TIME_STAMP = 786242.0
# eval 6906652  train 22211369 for 0.8
# eval 10273021 train 18845000 for 0.7
# eval 3544819  train 25573202 for 0.9
# eval 5254272  train 23863749 for 0.85 (roughly 22%)

- Based on Many scripts shared on forums found that many were using kNN and this was performing much better than Random forrest. So started to use kNN and started on tweaking features to obtain good results
# Total Eval Samples 7011, total classes 1023
# RFC (RandomForest) / KNC (KNearestNeighbor)
# (7 Features) top 3241.0 2878.0 2nd 751.0 1011.0 top3 4348.0 4328.0 no idx 539.0 539.0 top10 4971.0 5189.0 top20 5346.0 5513.0
# (3 Features) top 2646.0 2780.0 2nd 945.0 1002.0 top3 4055.0 4277.0 no idx 539.0 539.0 top10 4979.0 5178.0 top20 5300.0 5503.0 top30 5342.0 5577.0
# Except top in all positions KNN performs better than Random Forest
# KNN performs almost same with just three features instead of 7 features
# top3 gives 61%, top10 73.85% Accuracy, top20  78.49% Accuracy, top30 seems to achieve 79.54%

# KNN
# (n_neighbors=100)  top1,2,3 2801.0 971.0  4263.0 top-1,10,20,30 539.0 5182.0 5516.0 5582.0
# (n_neighbors=200)  top1,2,3 2790.0 1000.0 4280.0 top-1,10,20,30 539.0 5214.0 5574.0 5716.0
# (n_neighbors=1000) top1,2,3 2701.0 1002.0 4219.0 top-1,10,20,30 539.0 5133.0 5616.0 5796.0

# with 4 features ((data_element[4]%(60*24))*10)
# (n_neighbors=1000) top1,2,3 2629.0 790.0 3831.0 top-1,10,20,30 539.0 4798.0 5269.0 5518.0
# (data_element[4]%(60*24))
# (n_neighbors=1000) top1,2,3 2934.0 878.0 4239.0 top-1,10,20,30 539.0 5174.0 5593.0 5763.0
# (n_neighbors=500) top1,2,3 3050.0 889.0 4390.0 top-1,10,20,30 539.0 5263.0 5625.0 5766.0
# (n_neighbors=100) top1,2,3 3202.0 931.0 4536.0 top-1,10,20,30 539.0 5287.0 5525.0 5601.0
# without log scaling of accuracy
# (n_neighbors=100) top1,2,3 3210.0 887.0 4511.0 top-1,10,20,30 539.0 5263.0 5533.0 5618.0
# without weights=calculate_distance and metric='manhattan'
# (n_neighbors=100) top1,2,3 3114.0 894.0 4444.0 top-1,10,20,30 539.0 5248.0 5509.0 5589.0

# with time offset 18*
# top1,2,3 3229.0 870.0 4514.0 top-1,10,20,30 539.0 5272.0 5531.0 5625.0

Not logged the summary of things done within analysing_erros.py and matching_script3.py
The kaggle scripts used have the link to the original forums where the idea are discussed in much better fashion

- Then started to go in detail by looking into one file and trying to find the histogram of errors i.e. finding out which particular place id did not match the expected result.
- Also, then started to make 2-nd best value histogram and trying to make it the best.
- Observed graphs and scatter plots of the data distribution for these specific cases and tried to find out the exact difference
- This led to the Hour in Day feature and using sin and cos to encode it
- In this process tried evaluating KDE based classifiers - Because found that the data distribution seem like gausian for accuracy and it seem to have multiple peaks
    - To model this tried to use KDE but was slow and not able to combine the cost of KDE with KNN
- Check out the analysing_errors.py file for more info

- Then ran a MADZ kaggle script which gave a very good performance. Tried to match my script to this script. checkout matching_script3.py for more info



 