# Create N augmented files for each category
#   Src: D:\Visible_Data\3.SplitTrainVal\[Train|Val]\[1|2|3|4|m|ma]\* (listed in ListLabelledFiles.csv)
#   Dest: D:\Visible_Data\4.Augmented\[Train|Val]\[1|2|3|4|m|ma]\<origfilename>_counter.[png|jpg]


import math
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os

save_to_dir_template = r"D:\Visible_Data\4.Augmented"
batch_size = 64

train_or_tests = ["Train","Val"]

# Feedback 2020.02.24: create #augmented_samples for each class = max #samples in orig class
files_to_augment_per_class = {"Train": 1548, "Val": 387}
#files_to_augment_per_class = {"Train": 8000, "Val": 2000}

subcategories = ["1","2","3","4","m","ma"]


print ('Reading file ListLabelledFiles.csv...')
df_files = pd.read_csv ('ListLabelledFiles.csv', header=None, names=["train_or_test","subcategory","filepath"])
print ('Done Reading')

datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=32,
    height_shift_range=32,
    #brightness_range=[0.,2.],
    zoom_range=0.1,
    horizontal_flip=True
)


for cur_train_or_test in train_or_tests:

    # Remove destination dir
    Test_or_Train_Root = "\\".join([save_to_dir_template, cur_train_or_test])
    if os.path.isdir( Test_or_Train_Root ):
        shutil.rmtree(Test_or_Train_Root)
    os.mkdir(Test_or_Train_Root)

    for cur_subcategory in subcategories:

        df_files_cur = df_files [ df_files.train_or_test.eq( cur_train_or_test ) & df_files.subcategory.eq( cur_subcategory ) ]

        save_to_dir = "\\".join([save_to_dir_template, cur_train_or_test, cur_subcategory])

        # Create destination dir
        os.mkdir(save_to_dir)

        # Init how many files augmented for the cur_subcategory
        files_agmented_cur_subcategory = 0

        augmenter=datagen.flow_from_dataframe(dataframe=df_files_cur, x_col="filepath", y_col="subcategory",
                                              class_mode="categorical", target_size=(255,255),
                                              save_to_dir= save_to_dir , save_format="jpg", save_prefix="",
                                              batch_size=batch_size, shuffle=False)

        #for batch_id in range(batches_per_class):
        while files_agmented_cur_subcategory < files_to_augment_per_class[cur_train_or_test]:
            X,y = augmenter.next()
            files_agmented_cur_subcategory += y.shape[0]
            print ("Class {0}, augmented {1} of {2}".format ( cur_subcategory, files_agmented_cur_subcategory, files_to_augment_per_class[cur_train_or_test] ) )