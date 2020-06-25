# Predicts each file from ListValFiles.csv
#   Ouputs errors to %GDRIVE%\PhD_Data\Visible_ErrorAnalysis\Misclassifications\[true_class]\[predicted_class].orig_filename.[jpg|png]

from tensorflow.keras.models import load_model
import pandas as pd
import os
import numpy as np
import shutil
from PIL import Image

save_to_dir_template = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Misclassifications"

# Load model
model_filename = "j:\\Visible_models\\6class\\model_6classes_v60.h5"
model = load_model(model_filename)
target_shape = ( model.layers[0].input_shape[1], model.layers[0].input_shape[2] )

# Prep validation file names
df_files = pd.read_csv ('ListValFiles.csv', header=None, names=["subcategory","filepath"])


# Remove and create destination dirs
subcat_names = np.unique (df_files.subcategory).tolist()
for subcat_level1 in subcat_names:
    subcat1_root = "\\".join([save_to_dir_template, subcat_level1])
    if os.path.isdir(subcat1_root):
        shutil.rmtree(subcat1_root)
    os.mkdir(subcat1_root)
    for subcat_level2 in subcat_names:
        if subcat_level1 != subcat_level2:
            subcat1_subcat2_root = "\\".join([subcat1_root, subcat_level2])
            os.mkdir(subcat1_subcat2_root)
print ("Recreated target dirs in ",save_to_dir_template)

# Predict test files; save erroroneous guesses
cnt_errors = 0
for i,row in df_files.iterrows():
    img = Image.open( row.filepath ).resize( target_shape )
    imgs_arr = np.asarray ( img )[np.newaxis] / 255.# change type and and 0h axis (#samples)
    pred = model.predict (imgs_arr)
    pred_subcat_ind = np.argmax (pred)
    if subcat_names.index(row.subcategory) != pred_subcat_ind:
        # Save erroroneous guess
        filename = row.filepath.split("\\")[-1]
        error_filepath = "\\".join ( [ save_to_dir_template, row.subcategory, subcat_names[pred_subcat_ind], filename] )
        img.save( error_filepath )
        #print ("error_filepath: ",error_filepath)
        cnt_errors += 1
    if i%100 == 0:
        print ("Processed {0} files".format(i))

print ("Saved {0} errors".format(cnt_errors))