import Model_6class.Train_6classes_v1 as t_6_v1
import numpy as np

# Batch norm layers
bn_layers_list = {
    14: ["d-2"],                    #v14
    15: ["d-2","d-3"],              #v15
    16: ["d-2","d-3","c+4"],         #v16
    17: ["d-2","d-3","c+3","c+4"],    #v17
    18: ["c+3","c+4"],                #v18
    19: ["c+1", "c+2", "c+3", "c+4","d-2","d-3"]  # v19
}

# Dropout layers
dropout_layers_list = {
    20: ["d-2"],
    21: ["d-2","d-3"]
}

# L2 regularization layers
l2_layers_list = {
    22: {"d-2": 0.1},
    23: {"d-2": 0.1, "d-3": "0.1"},
    24: {"d-2": 0.5, "d-3": "0.5"},
}

for i in np.arange(19,20):
    model = t_6_v1.trainModel(epochs=100,
                              bn_layers=bn_layers_list[i] if i in bn_layers_list else [],
                              dropout_layers=dropout_layers_list[i] if i in dropout_layers_list else [],
                              l2_layers = l2_layers_list[i] if i in l2_layers_list else {})
    model_file_name = r"J:\Visible_models\model_6classes_v" + str(i) + ".h5"
    #model.save(model_file_name)
    del model

#model = load_model (model_file_name)

#exec('Model_6class.conf_matrix_6classes.py')