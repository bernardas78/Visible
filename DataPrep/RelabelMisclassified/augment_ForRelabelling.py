# Create N augmented files for each category
#   Src: D:\Visible_Data\2.Cropped\[1|2|3|4|m|ma]\* (listed in ListLabelledUnsplit_ForAugmentation\[1|2|3|4|m|ma].csv)
#   Dest: D:\Visible_Data\AugmentedForRelabellingMisclassified\[1|2|3|4|m|ma]\<origfileindex>_counter.[png|jpg]


import math
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os

save_to_dir_template = r"D:\Visible_Data\AugmentedForRelabellingMisclassified"
batch_size = 64

# Feedback 2020.02.24: create #augmented_samples for each class = max #samples in orig class
files_to_augment_per_class = 1923

subcategories = ["1","2","3","4","m","ma"]

datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=32,
    height_shift_range=32,
    #brightness_range=[0.,2.],
    zoom_range=0.1,
    horizontal_flip=True
)

for cur_subcategory in subcategories:

    subcat_filenames_file = 'ListLabelledUnsplit_ForAugmentation\\' + cur_subcategory + '.csv'
    print( 'Reading file ' + subcat_filenames_file )
    df_files_cur = pd.read_csv(subcat_filenames_file, header=None, names=["filepath"])
    print('Done Reading')

    save_to_dir = "\\".join([save_to_dir_template, cur_subcategory])

    # Re-Create destination dir
    if os.path.isdir(save_to_dir):
        shutil.rmtree(save_to_dir)
    os.mkdir(save_to_dir)

    # Init how many files augmented for the cur_subcategory
    files_agmented_cur_subcategory = 0

    augmenter=datagen.flow_from_dataframe(dataframe=df_files_cur, x_col="filepath",
                                          class_mode=None, target_size=(255,255),
                                          save_to_dir= save_to_dir , save_format="jpg", save_prefix="",
                                          batch_size=batch_size, shuffle=False)

    #for batch_id in range(batches_per_class):
    while files_agmented_cur_subcategory < files_to_augment_per_class:
        X = augmenter.next()
        files_agmented_cur_subcategory += X.shape[0]
        print ("Class {0}, augmented {1} of {2}".format ( cur_subcategory, files_agmented_cur_subcategory, files_to_augment_per_class ) )