# Create N augmented files for each category
#   Src: D:\Visible_Data\3.SplitTrainVal\[Train|Val]\[1|2|3|4|m|ma]\* (listed in ListLabelledFiles.csv)
#   Dest: D:\Visible_Data\4.Augmented\[Train|Val]\[1|2|3|4|m|ma]\<origfilename>_counter.[png|jpg]


import math
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os
import numpy as np

save_to_dir_template = r"D:\Visible_Data\4.Augmented"
batch_size = 32

sets = ["Train","Val","Test"]

print ('Reading file ListLabelledFiles.csv...')
df_files = pd.read_csv ('ListLabelledFiles.csv', header=None, names=["train_or_test","subcategory","filepath"])
print ('Done Reading. Shape: ' + str(df_files.shape))


# Feedback 2020.02.24: create #augmented_samples for each class = max    #samples in orig class
#files_to_augment_per_class = {"Train": 8000, "Val": 2000}               #initial logic: just a lot
#files_to_augment_per_class = {"Train": 1548, "Val": 387 }               # per feedback: up to max class
#files_to_augment_per_class = {"Train": 1200, "Val": 300, "Test": 375}   # adding test set
#files_per_class = {"Train": 1200, "Val": 300, "Test": 0}     # starting v55, do not augment/balance test data
files_per_class = {"Train": df_files[df_files.train_or_test=="Train"].groupby(['subcategory']).count().max()[0],  # calculate: up to max class
                   "Val": df_files[df_files.train_or_test=="Val"].groupby(['subcategory']).count().max()[0],
                   "Test": 0} # do not augment Test class

# Experiments with dataset size
#files_per_class = {"Train": 1100, "Val": 275, "Test": 0}     # starting v55, do not augment/balance test data
#files_per_class = {"Train": 1000, "Val": 250, "Test": 0}     # starting v55, do not augment/balance test data
#files_per_class = {"Train":  900, "Val": 225, "Test": 0}     # starting v55, do not augment/balance test data

#subcategories = ["1","2","3","4","m","ma"]
subcategories = np.unique ( df_files.subcategory).tolist() #make sure it works for 2 or 6 classes



datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=32,
    height_shift_range=32,
    zoom_range=0.1,
    horizontal_flip=True
)


for cur_set in sets:

    # Remove destination dir
    Test_or_Train_Root = "\\".join([save_to_dir_template, cur_set])
    if os.path.isdir( Test_or_Train_Root ):
        shutil.rmtree(Test_or_Train_Root)
    os.mkdir(Test_or_Train_Root)

    for cur_subcategory in subcategories:

        df_files_cur = df_files [ df_files.train_or_test.eq( cur_set ) & df_files.subcategory.eq( cur_subcategory ) ]

        save_to_dir = "\\".join([save_to_dir_template, cur_set, cur_subcategory])

        # First, copy original files to dest
        print ("Copying original files in {}, sucbcat {}".format (cur_set, cur_subcategory) )
        src_subcat_dir = "\\".join ( df_files_cur.iloc[0].filepath.split("\\")[:-1] ) # remove filename
        shutil.copytree( src_subcat_dir, save_to_dir)
        files_copied = len ( os.listdir(save_to_dir) )
        print("Done Copying {} original files".format(files_copied) )

        # Create destination dir (no longer needed since copying original files takes care of that
        #os.mkdir(save_to_dir)

        # Init how many files augmented for the cur_subcategory (originals included)
        files_agmented_cur_subcategory = files_copied

        #print('Before flow_from_dataframe. Shape: ' + str(df_files_cur.shape))
        augmenter=datagen.flow_from_dataframe(dataframe=df_files_cur, x_col="filepath", y_col="subcategory",
                                              class_mode="categorical", target_size=(256,256),
                                              save_to_dir= save_to_dir , save_format="jpg", save_prefix="",
                                              batch_size=batch_size, shuffle=False)

        while files_agmented_cur_subcategory < files_per_class[cur_set]:
            X,y = augmenter.next()
            files_agmented_cur_subcategory += y.shape[0]
            print ("Class {0}, augmented {1} of {2}".format ( cur_subcategory, files_agmented_cur_subcategory, files_per_class[cur_set] ) )