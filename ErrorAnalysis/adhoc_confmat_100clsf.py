# Confusion matrix
#   Input: Result of 100 classifiers: D:\Google Drive\PhD_Data\Visible_ErrorAnalysis\Relabelling\Preds100Clsf.csv
#   Outs: conf matrix (w, w/o diagonal)

import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

filename_100clsf_res = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\Preds100Clsf.csv"

conf_mat_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Conf_Mat\\model_6classes_100clsf_conf_mat.png"
conf_mat_no_diag_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Conf_Mat\\model_6classes_100clsf_conf_mat_no_diag.png"

subcategory_names = ["1","2","3","4","m","ma"]

print ("Loading Preds100Clsf.csv...")
df_100clsf_res = pd.read_csv ( filename_100clsf_res, header=0)
print ("Loaded Preds100Clsf.csv")

y_true=df_100clsf_res.actual #/ len (df_100clsf_res.actual)  #+ "%"
y_pred=df_100clsf_res.predicted # /len(df_100clsf_res.predicted)  #+ "%"

conf_mat = confusion_matrix(y_true=y_true,y_pred=y_pred)

# Display percentages, since numbers too big
total_cnt = np.sum(conf_mat)
conf_mat_formatted = np.round ( conf_mat/total_cnt*100, 1 )

ax = sns.heatmap( conf_mat_formatted, annot=True, fmt='g' )
ax.set_xticklabels( subcategory_names, horizontalalignment='right' )
ax.set_yticklabels( subcategory_names, horizontalalignment='right' )
plt.savefig (conf_mat_file_name)
plt.close()

# No diagonal confusion matrix
for i in range(conf_mat_formatted.shape[0]):
    conf_mat_formatted[i,i] = 0
ax = sns.heatmap( conf_mat_formatted, annot=True, fmt='g' )    #/np.sum(conf_mat)
ax.set_xticklabels( subcategory_names, horizontalalignment='right' )
ax.set_yticklabels( subcategory_names, horizontalalignment='right' )
plt.savefig (conf_mat_no_diag_file_name)
plt.close()
