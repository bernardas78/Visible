# function to ready directory contents and split into train and validation dataframes

from sklearn.utils import shuffle
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# Root folder of images
# Manually copied to C: to speed up training
# data_dir_6classes = r"D:\Visible_Data\AugmentedForRelabellingMisclassified"
data_dir_6classes = r"C:\AugmentedForRelabellingMisclassified"

# Validation data percentage
val_pct = 0.2

subcategory_names = ["1","2","3","4","m","ma"]

def splitTrainVal ():

    pic_filenames = []
    pic_subcategories = []
    for r, d, f in os.walk(data_dir_6classes):
        for file in f:
            pic_filenames.append(os.path.join(r, file))

            subcategory_name = r.split(sep="\\")[-1]
            subcategory = subcategory_names.index( subcategory_name )
            pic_subcategories.append( subcategory )

    pic_IsTrain = np.repeat(1, len(pic_filenames))

    # Sse 1st fold of K-fold CV to split Train/Val
    skf = StratifiedKFold ( shuffle=True, n_splits= int(1/val_pct) )
    for train_index, test_index in skf.split(pic_filenames, pic_subcategories):
        pic_IsTrain[test_index] = 0
        #we just need 1st fold
        break

    column_names = ['isTrain', 'subcategory', 'filepath']
    df_all = pd.DataFrame(columns=column_names, data=np.asarray((pic_IsTrain, pic_subcategories, pic_filenames)).T)
    # Training does not converge unless shuffled
    df_all = shuffle(df_all)

    df_train = df_all.loc[df_all.isTrain=="1"].drop("isTrain",1)
    df_val = df_all.loc[df_all.isTrain=="0"].drop("isTrain",1)

    return df_train, df_val