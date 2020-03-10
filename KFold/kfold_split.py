# Reads file names and makes a single file (later used in AugSequence's CV)
# To run:
#   cd C:\labs\KerasImagenetFruits\PreprocessImages
#   python
cv_folds = 5
picDir = "C:\\KFold_Visible\\"
picnamesFile = str(cv_folds) + "_filenames.csv"

import os
import pickle
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np



pic_filenames = []
pic_subcategories = []
for r, d, f in os.walk(picDir):
    for file in f:
        pic_filenames.append (os.path.join(r, file) )
        pic_subcategories.append ( r.split(sep="\\")[-1] )
pic_folds = np.repeat ( -1, len(pic_filenames) )


# K-fold CV
fold_id = 0
skf = StratifiedKFold(n_splits=cv_folds)
for train_index, test_index in skf.split(pic_filenames, pic_subcategories):
    pic_folds[test_index] = fold_id
    fold_id += 1

column_names = ['fold','subcategory','filepath']
df_kfold = pd.DataFrame (columns=column_names, data = np.asarray ( (pic_folds, pic_subcategories, pic_filenames ) ).T )
df_kfold.to_csv(picnamesFile, index=False, header=True, mode='w')

