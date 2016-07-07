


TRAIN_FILE_PATH = "../data/train.csv"
TEST_FILE_PATH = "../data/test.csv"


TRAIN_ALL_INT_PATH = "../data/common/train_int.csv"
TEST_ALL_INT_PATH = "../data/common/test_int.csv"

SPLIT_TRAIN_ALL_INT_PATH = "../data/common/split_train.csv"
SPLIT_EVAL_ALL_INT_PATH = "../data/common/split_eval.csv"


UNIQUE_ID_PATH = "../data/common/uni_id.csv"
STATS_PATH = "../data/common/stats.csv"
SPLIT_STATS_PATH = "../data/common/split_stats.csv"
REC_ID_TO_FIL_ID_NAME = "record_id_to_file_id.csv"



# split file based on y value
Y_BASED_SPLIT_TRAIN_PATH = "../data/y_based_split/train/ybs_tr"
Y_BASED_SPLIT_FULL_TRAIN_PATH = "../data/y_based_split/full_train/ybs_ftr"
Y_BASED_SPLIT_EVAL_PATH = "../data/y_based_split/eval/ybs_ev"
Y_BASED_SPLIT_TEST_PATH = "../data/y_based_split/test/ybs_te"



#pred_ne10_nd18
# Y_BASED_SPLIT_EVAL_PRED_PATH = "../data/y_based_split/eval/pred_ne10_nd18/ybs_ev_pr"
# Y_BASED_SPLIT_EVAL_PRED_TN_PATH = "../data/y_based_split/eval/pred_ne10_nd18/topn/ybs_ev_pr_tn"
# Y_BASED_SPLIT_EVAL_PRED_TNM_PATH = "../data/y_based_split/eval/pred_ne10_nd18/topn/ybs_ev_pr_tnm.csv"
Y_BASED_SPLIT_EVAL_PRED_PATH = "../data/y_based_split/eval/pred/ybs_ev_pr"
Y_BASED_SPLIT_EVAL_PRED_TN_PATH = "../data/y_based_split/eval/pred/topn/ybs_ev_pr_tn"
Y_BASED_SPLIT_EVAL_PRED_TNM_PATH = "../data/y_based_split/eval/pred/topn/ybs_ev_pr_tnm.csv"


Y_BASED_SPLIT_TEST_PRED_PATH = "../data/y_based_split/test/pred/ybs_te_pr"
Y_BASED_SPLIT_TEST_PRED_TN_PATH = "../data/y_based_split/test/pred/topn/ybs_te_pr_tn"
Y_BASED_SPLIT_TEST_PRED_TNM_PATH = "../data/y_based_split/test/pred/topn/ybs_te_pr_tnn.csv"



TRAINING_PERCENT = 0.8 #Evaluation percent 0.2

Y_STEP_SIZE = 1500#0.15*10000
#Y_RANGE = 4500#0.45*10000
MAX_XY_VALUE = 100000


XY_SD_SP_TR80_PATH = "../data/xy_sd_split/train80/xys_tr"
XY_SD_SP_TR80_3SD_PATH = "../data/xy_sd_split/train80_3sd_10000x_1500y/xys_tr"
XY_SD_SP_TR80_2SD_PATH = "../data/xy_sd_split/train80_2sd_10000x_1500y/xys_tr"
XY_SD_SP_TR_TS_2SD_PATH = "../data/xy_sd_split/train_ts_2sd_10000x_1500y/xys_tr"


XY_SD_SP_EVAL_PATH = "../data/xy_sd_split/eval/xys_ev"
XY_SD_SP_EVAL_3SD_PATH = "../data/xy_sd_split/eval_3sd_10000x_1500y/xys_ev"

XY_SD_SP_EVAL_2SD_PATH = "../data/xy_sd_split/eval_2sd_10000x_1500y/xys_ev"
XY_SD_SP_EVAL_PRED_2SD_PATH = "../data/xy_sd_split/eval_2sd_10000x_1500y/pred/xys_ev_pr"
XY_SD_SP_EVAL_PRED_TNM_2SD_PATH = "../data/xy_sd_split/eval_2sd_10000x_1500y/pred/topn/xys_ev_pr_tnm.csv"

XY_SD_SP_EVAL_TS_2SD_PATH = "../data/xy_sd_split/eval_ts_2sd_10000x_1500y/xys_ev"
XY_SD_SP_EVAL_TS_2SD_PRED_PATH = "../data/xy_sd_split/eval_ts_2sd_10000x_1500y/pred/xys_ev_pr"
XY_SD_SP_EVAL_TS_2SD_PRED_TNM_PATH = "../data/xy_sd_split/eval_ts_2sd_10000x_1500y/pred/topn/xys_ev_pr_tnm.csv"



XY_SD_SP_FUTR_PATH = "../data/xy_sd_split/full_train/xys_ft"
XY_SD_SP_TEST_PATH = "../data/xy_sd_split/test/xys_te"
XY_SD_SP_TEST_PRED_PATH = "../data/xy_sd_split/test/pred/xys_te_pr"
XY_SD_SP_TEST_PRED_TNM_PATH = "../data/xy_sd_split/test/pred/topn/xys_te_pr_tnm.csv"



# Based on std deviation (sigma) 1sigma, 2sigma, 3sigma
# 68-95-99.7 rule
# Average 1-sd (std deviation) implies 70% of values would lie between +/- 1sd
# 108391 6750.0715741 206.229965204 106.961287001 (num_entries, x, y, acc)
# Average 2-sd imples 95% values lie between +/- 2sd
# 108391 13500.1431482 412.459930408 213.922574002
# Average 3-sd (std deviation) implies 99% of values would lie between +/- 3sd
# 108391 20250.2147223 618.689895612 320.883861004
# Y_STEP = 619 # 413 # 206
# X_STEP = 20250 # 13500 # 6750
Y_RNG = 1500
X_RNG = 10000
NUM_SD = 2





