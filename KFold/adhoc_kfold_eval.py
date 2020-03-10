from KFold import  kfold_Metrics as m_kfold_v1
#from Eval import PredsToFile as ptf
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import scipy.stats
import numpy as np

cv_folds = 5

start_fold = 0

invisible_subcategories=["1"]

metricsFile = "D:\\Visible\\Eval\\" + str(cv_folds) + "fold_eval_v1.csv"

#column_names = ["accuracy", "precision", "recall", "f1"]
df_metrics = pd.DataFrame()         #.to_csv(metricsFile, index=False, header=True, mode='w') columns=column_names

for fold_id in range(start_fold, cv_folds):
    model_file_name = "J:\\Visible_models\\model_2classes_fold_" + str(fold_id) + "_of_" + str(cv_folds) + "_v1.h5"
    kfold_csv = str(cv_folds) + "_filenames.csv"

    model = model = load_model (model_file_name )
    metrics = m_kfold_v1.CalcMetrics(model=model, kfold_csv=kfold_csv,  fold_id=fold_id, invisible_subcategories=invisible_subcategories)
    df_metrics = df_metrics.append( metrics, ignore_index=True )

df_metrics.to_csv(metricsFile, header=None, index=None, mode='w')

# Calculate which mu value the accuracy is greater than with 95% confidence
t_stat = scipy.stats.t.ppf ( q = 0.95, df = cv_folds-1)
mu_sample = np.mean ( df_metrics.accuracy )
sigma_sample = np.std ( df_metrics.accuracy )
n = cv_folds
mu = mu_sample - t_stat * sigma_sample / np.sqrt(n)
print ("Sample Mu: {0}, Sigma: {1}, df: {2}".format(mu_sample, sigma_sample, n))
print ("T.statisctic, 95% confidence: {0} ".format(t_stat))
print ("Mu of population greater than {0}, 95% confidence".format(mu))