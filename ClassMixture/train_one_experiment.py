# Runs a single experiment (params: class mixture)

import shutil
import os
import ClassMixture.Train_ClassMixture_v1 as t_cm_v1
import ClassMixture.create_dataset as cd
from tensorflow.keras import backend as K

# Source folder with augmented 6-class images
src_data_folder = "D:\\Visible_Data\\4.Augmented"
# Data folder of 2-class images which will be recreated
dest_data_folder = "C:\\TrainAndVal"
# Where to save trained models
model_path_pattern = r"J:\ClassMixture_Models\model_v"



def trainSingleClassMixtureExperiment (Subcats, version):
    # Subcats: dictionary {"Visible: ["4",..], "Invisible: []}
    print ("Running experiment vis/invis {}/{}".format(Subcats["Visible"], Subcats["Invisible"]))

    # Create dataset with proper mix of subcats into Visible/Invisible
    cd.create_dataset (Subcats, version)

    # v58, v60
    params_dict = {"epochs": 100,
                   "bn_layers": ["c+1", "c+2", "c+3", "c+4", "c+5", "c+6", "c+7", "c+8", "d-2", "d-3"],
                   "dropout_layers": ["d-2", "d-3"],
                   "l2_layers": {},
                   "padding": "same",
                   "target_size": 256,
                   "dense_sizes": {"d-3": 256, "d-2": 128, "d-1": 2},
                   "architecture": "Model_classmixture_v1",
                   "conv_layers_over_5": 2,
                   "use_maxpool_after_conv_layers_after_5th": [False, True],
                   "version": version,
                   "load_existing": True}
    model = t_cm_v1.trainModel(**params_dict)
    model_file_name = model_path_pattern + str(version) + ".h5"
    #model.save(model_file_name)
    del model
    K.clear_session()


    # Evaluate model