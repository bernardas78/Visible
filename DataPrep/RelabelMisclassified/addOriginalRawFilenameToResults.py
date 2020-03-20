# Take Preds100Clsf.csv (classification result of 100 classifiers)
# Add original (Raw) filename - save file Preds100Clsf_IncRawFilename.csv

import pandas as pd
from os import path
import os

filename_100clsf_res = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\Preds100Clsf.csv"
filename_100clsf_res_iclOrig = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\Preds100Clsf_IncRawFilename.csv"
orig_filenames_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Relabelling\\ListLabelledUnsplit_ForAugmentation\\"
raw_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Raw\\"
scos = ["SCO1","SCO2","SCO3","SCO4"]

# Load all subcategory files for lookup of filename
subcategory_names = ["1","2","3","4","m","ma"]
subcategory_dfs = {}
for subcategory_name in subcategory_names:
    subcategory_orig_filenames_file = orig_filenames_folder + subcategory_name + ".csv"
    subcategory_df = pd.read_csv (subcategory_orig_filenames_file, header=None)
    subcategory_dfs.update ( {subcategory_name: subcategory_df} )

df_100clsf_res = pd.read_csv ( filename_100clsf_res, header=0)

# add column for original filename
df_100clsf_res["orig_filename"] = "NOT_FOUND"

for index,row in df_100clsf_res.iterrows():
    #print (row.filename)

    # Parse class and file index from structure ...\class\_<fileind>_....jpg
    fileind = int ( row.filename.split("\\")[-1].split("_")[1] )
    fileclass = row.filename.split("\\")[-2]
    if index==0:
        print ("File index:{0} in class: {1}".format(fileind, fileclass))

    # Lookup filename in ListLabelledUnsplit_ForAugmentation\<class>.csv
    orig_filename = subcategory_dfs[fileclass].loc[fileind, 0]
    orig_filename_nopath = orig_filename.split("\\")[-1]
    if index==0:
        print ("Orig filename {0}, with path {1}".format(orig_filename_nopath,orig_filename) )

    # Find in "Raw" folder (search in 4 difference SCOs)
    found = False
    for sco in scos:
        potential_filename = raw_folder + sco + "\\" + fileclass + "\\" + orig_filename_nopath
        if index==0:
            print ("Looking for file:{0}".format(potential_filename) )

        if path.exists(potential_filename):
            if index == 0:
                print("Found file:{0}".format(potential_filename))
            found=True
            df_100clsf_res.loc[index, "orig_filename"] = potential_filename

    if not found:
        print ("File {0} not found".format(orig_filename_nopath))



    if index%1000==0:
        print ("Processed {0}".format(index))
    #break

df_100clsf_res.to_csv(filename_100clsf_res_iclOrig, index=None)