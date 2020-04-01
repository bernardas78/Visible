import Model_2class.Train_2classes_v1 as t_2_v1
import numpy as np

start_version = 30
end_version = 30

# Network architecture
architecture_dic = [ *np.repeat("Model_6classes_c4_d3_v1", 28).tolist(), # v<=28
                     *np.repeat("Model_6classes_c5_d2_v1", 1).tolist(),  # v=29
                     *np.repeat("Model_6classes_c5_d3_v1", 1).tolist(),  # v=30
                     *np.repeat("Model_6classes_c6_d2_v1", 1).tolist(),  # v=31
                     *np.repeat("Model_6classes_c4_d3_v1", 100).tolist() ] # Revert back to ?v28? after 31

# Padding, target_size
padding_same_list = np.arange(27,end_version+1)
target_size_256_list = np.arange(27,end_version+1)

# Dense sizes
dense_sizes_dic = {
    "default_1": {"value": {"d-3": 2048, "d-2": 128}, "versions": list(range(28)) },    # v<=27
    "default_2": {"value": {"d-3": 256, "d-2": 128}, "versions": [28,30] },             # v=28, 30
    "default_3": {"value": {"d-2": 128}, "versions": [29, 31] }                         # v=29, 31
}


# Batch norm layers
bn_layers_list = {
    14: ["d-2"],                    #v14
    15: ["d-2","d-3"],              #v15
    16: ["d-2","d-3","c+4"],         #v16
    17: ["d-2","d-3","c+3","c+4"],    #v17
    18: ["c+3","c+4"],                #v18
    19: ["c+1", "c+2", "c+3", "c+4","d-2","d-3"],  # v19
    20: [], 21: [], 22: [], 23: [], 24: [],
    25: ["c+1", "c+2", "c+3", "c+4", "d-2", "d-3"],
    "default": ["c+1", "c+2", "c+3", "c+4", "c+5", "c+6", "d-2", "d-3"]
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

for i in np.arange(start_version,end_version+1):
    params_dict = {"epochs": 100,
                   "bn_layers": bn_layers_list[i] if i in bn_layers_list else bn_layers_list["default"],
                   "dropout_layers": dropout_layers_list[i] if i in dropout_layers_list else [],
                   "l2_layers": l2_layers_list[i] if i in l2_layers_list else {},
                   "padding": "same" if i in padding_same_list else "valid",
                   "target_size": 256 if i in target_size_256_list else 224,
                   "dense_sizes": dense_sizes_dic["default_1"]["value"]
                                        if i in dense_sizes_dic["default_1"]["versions"]
                                        else dense_sizes_dic["default_2"]["value"]
                                            if i in dense_sizes_dic["default_2"]["versions"]
                                            else dense_sizes_dic["default_3"]["value"],
                   "architecture": architecture_dic[i-1]}
    model = t_2_v1.trainModel( **params_dict )
    model_file_name = r"J:\Visible_models\2class\model_2classes_v" + str(i) + ".h5"
    model.save(model_file_name)
    del model

#model = load_model (model_file_name)

#exec('Model_6class.conf_matrix_6classes.py')