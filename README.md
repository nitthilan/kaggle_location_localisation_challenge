Learnings on Facebook Predicting Check Ins Competition:
=======================================================
[To be continued from analysing_erros.py and matching_script3.py experiments data. Currently refer the file for more information]

Things done in chronological order:
- Wrote a script to understand the data (did only for train but better to do fo both train and test)
- Essential things to note down:
    - Num entries, Range of each input (min, max, mean, std. deviation)
    - How the scoring is done
    - Num entries in each class/category
    - Num unique categories

- Understanding data is line Accuracy, Time feature
    - What individual data means Accuracy, time Feature
    - Visualising data (Distribution of data)
        - How many entries are available for each place_id - histogram of num entries
        - Individual fields distributed in a graph
        - Relation between data X, Y, Accuracy, Time (Scatter plots)
        - Mean/Std Deviation values for each feature for different place ids
- Preprocessing of Data: (May or may not work)
    - Normalising the data (mapping to 0-1 based on range of data (num decimals to represent the data))
    - convert everything to integers
    - Mean/Std. Deviation of features for individual classes (gave a feel how data was distributed)
- Create a test setup:
    - Split the data as Train/Validation
    - Have two setups: 
        - Train split as 80% Train Vs 20% Eval for validating perfromance
        - Full Train set Vs Test set (for final submission)
    - Have a downsampled test setup where you can run algorithms under five minutes to just check the various models
- Trying to reduce the complexity of the data
    - Downsample the input. Be careful and make the downsampled set as close as possible to Test data set.
    - Mistake: Down sampled (split) data such that each set has equal number of entries from each class. however the test data was split based on time
        - A simple time based split would have been sufficient
    - Split entries into various small files which can be parallely executed and results finally collated
    - Mistake: Using Zip files to reduce the intermediate files. But did not help much since file reads where faster than processing
        - Use zip files for final output only
- Trying out various algorithms:
    - Mistake: Tensor flow based Neural Network for classification. Many iterative algorithms did not work (SVM etc) out of the box
    - Mistake: Tried a custom Nearest Neighbor algorithm i.e. stored the mean and std dev for each placeid and used that to identify the points which are close to this mean and std deviation
        - Later this seems very weirdly close to NaiveBayes Algorithm
    - RandomForrest seem to give the initial results because did not know how to weigh using Baysian Optimisation for Nearest Neighbor Algorithms 
        - How ever it consumed lot of memory and it grew exponentially with the number of estimators (tried my luck with optimising based on tree depth, etc
    - First Proper result: Used Random forest. 
        - Problem: The number of training samples was in the order of 29118022 and num classes was 108390.
        - Observed that the distribution of samples for the Y feature had a very small std. deviation. So split data along the Y direction to around 60 rows
        - Now running random forrest on each of these rows was possible with num_estimators around 25
        - However there could be sample which cross boundaries of the splits and may lose on prediction 
        - First run took around 16 Hours
    - Next Tried to optimise on the various Hyper parameters of Random Forest (See the results page for more details)
    - Mistake: Increased the feature set by adding time features since it was observed that the time value was in minutes
        - Have increased number of features from usual x,y, accuaracy to time based features like hour in a day, day in a week, day in a month, just day id, year etc
        - These features seem to give very good results on the eval set but when uploaded they provided a very bad result. 
        - there was a huge difference between evaluation results and actual results
        - This was because we did not split the eval data set based on time and have split it based on just file position, 
        - while the train data set specifically mentions this.
        - This helped in predicting the 
        - Split the data based on time and generated output.
    - To take care of the points across the boundary split thought of using 
        - a single classfier which combines three train splits (left, center and right) and trains the classifier
            - This caused the memory of Random forest to shoot up drastically So did not try
        - Mistake: a Run RF for left, center and right with the same eval/train set and merge the results based on the pred_prob values
            - Cannot combin the result of two separate trees trained on different data sets. 
                - This is because they are not normalised across all the data sets
                - A pred value could have more close estimates in one data set while could have 
    - Next calcualted the mean and std. deviation for every place_id and then used the mean value to choose whether the particular candidate was part of the split file or not
        - The Eval/Train file used the actual y features
        - Random forest could run without issues
        - Based on the Assumption that 70-95-99 rule of std deviation for 1sd-2sd-3sd
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
    - Based on Many scripts shared on forums found that many were using kNN and this was performing much better than Random forrest. So started to use kNN and started on tweaking features to obtain good results
    - To be continued from analysing_erros.py and matching_script3.py experiments data. Currently refer the file for more information

- Measure the running time (approx)
Things to Learn:
- Hadoop kind of systems HDFS/Spark
- Multiprocessing to execute in threads parallely
Useful software:
- Matplotlib, pandas, time, scikit
Things learnt:
- Parse the data and make a description of the Test and Train data as soon as possible
- Make your first submission as soon as possible so that it validates your entire development process 
    - like is the way of evaluating the output, way or writing output files, having all the output entries are fine
    - it need not perform well in performance or algorithm wise but what you predicted and what you got should match
- Have a very fast Eval setup where you can iterate and validate algorithms you want to try
- Create a setup where you can fit the output of learnings of a Eval setup and do a submission (decently fast)
- Be very careful when you downsample a test/eval data set. Try to make the eval set as close as possible to how the train set was created.
- Use the Forum board and scripts shared by others. Visit it regularly and run the shared scripts blindly to learn new things
- Probably use Bayesian Optimisation to tune Hyperparameters. But initially, using values spread across in log domain 1, 10 ,100 and then refining it to smaller values like 10,11,12 ettc
- Try using multithreading to speed up execution
- Keep a check on the hardisk space (filled up the hard disk and found it hard to recover), execution times, how much resource you consume on the server processor memory, processor cpu cycles
    - How to keep the CPU fully engaged?
Unanswered Questions:
- why does dnn take lot of time to converge?
- why does SVM take lot of time to learn?
- why does random forrest consume lot of memeory?
- How to use num entries
- Like bayesian Optimisation for estimating Hyperparameters can we use Neural Networks to optimise or learn??
- How did downsampling based on number of samples produce a very high mean value of correct predictions???
- Using time based split and multiple models to get better predictions (based on Kaggle scripts6)

Things to do:
- Create a single script and submit it online for review
- Save all the files use in Github for future reference

Keywords:
- Eval Set, Train/Test Split, HyperParameters, Cross Validation, Feature Engineering