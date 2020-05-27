import Autoenc.Train_autoenc_v1 as t_autoenc_v1
#import Train_6classes_v1 as t_6_v1
import numpy as np

start_version = 2
end_version = 2

# Batch norm layers
bn_layers_list = ["c+1", "c+2", "c+3", "c+4", "c+5"]


for i in np.arange(start_version,end_version+1):
    params_dict = {"epochs": 100,
                   "bn_layers": bn_layers_list,
                   "target_size": 256,
                   "version": i}
    model = t_autoenc_v1.trainModel( **params_dict )
    model_file_name = r"J:\Visible_models\Autoenc\model_autoenc_v" + str(i) + ".h5"
    model.save(model_file_name)
    del model

#model = load_model (model_file_name)

#exec('Model_6class.conf_matrix_6classes.py')