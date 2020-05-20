import Model_6class.Train_6classes_v1 as t_6_v1
#import Train_6classes_v1 as t_6_v1
import numpy as np

start_version = 53
end_version = 53

# Network architecture
architecture_dic = [ *np.repeat("Model_6classes_c4_d3_v1", 28).tolist(), # v<=28
                     *np.repeat("Model_6classes_c5_d2_v1", 1).tolist(),  # v=29
                     *np.repeat("Model_6classes_c5_d3_v1", 1).tolist(),  # v=30
                     *np.repeat("Model_6classes_c6_d2_v1", 1).tolist(),  # v=31
                     *np.repeat("Model_6classes_c5_d3_v1", 1).tolist(),  # v=32
                     *np.repeat("Model_6classes_c5plus_d3_v1", 21).tolist(), # Parameterized #conv_layers for v33-53
                      *np.repeat("???", 100).tolist()
]

conv_layers_over_5_dic = {
    "default_3": {"value": 3, "versions": [35] },
    "default_2": {"value": 2, "versions": [34,37,38,39, 40, 41, 42, 43, 44, 45, 46, 47,48,49,50,51,52,53] },
    "default_1": {"value": 1, "versions": [33,36] },
    "default": {"value": 0, "versions": list(range(100)) },
}

use_maxpool_after_conv_layers_after_5th_dic = {
    33: [True],
    34: [True,True],
    35: [True,True,True],
    36: [False],
    37: [False,False],
    38: [False,True],
    39: [True,False],
    40: [False, True], 41: [False, True], 42: [False, True], 43: [False, True], 44: [False, True], 45: [False, True], 46: [False, True], 47: [False, True], 48: [False, True], 49: [False, True],
    50: [False, True], 51: [False, True], 52: [False, True], 53: [False, True]
}


# Padding, target_size
padding_same_list = np.arange(27,end_version+1)
target_size_256_list = np.arange(27,end_version+1)

# Dense sizes
dense_sizes_dic = {
    "default_1": {"value": {"d-3": 2048, "d-2": 128}, "versions": list(range(28)) },    # v<=27
    "default_2": {"value": {"d-3": 256, "d-2": 128}, "versions": [28,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53] },
    "default_3": {"value": {"d-2": 128}, "versions": [29, 31] }
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
    "default": ["c+1", "c+2", "c+3", "c+4", "c+5", "c+6", "c+7", "c+8", "d-2", "d-3"]     # v>=26
}

# Dropout layers
dropout_layers_list = {
    20: ["d-2"],
    21: ["d-2","d-3"],
    40: ["d-2"],
    41: ["d-3"],
    42: ["d-2", "d-3"],
    43: ["d-2"],44: ["d-2"],45: ["d-2"],46: ["d-2"],47: ["d-2"],48: ["d-2"],49: ["d-2"],50: ["d-2"],51: ["d-2"],52: ["d-2"],53: ["d-2"]
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
                   "architecture": architecture_dic[i-1],
                   "conv_layers_over_5": conv_layers_over_5_dic["default_3"]["value"]
                                            if i in conv_layers_over_5_dic["default_3"]["versions"]
                                            else conv_layers_over_5_dic["default_2"]["value"]
                                                if i in conv_layers_over_5_dic["default_2"]["versions"]
                                                else conv_layers_over_5_dic["default_1"]["value"]
                                                    if i in conv_layers_over_5_dic["default_1"]["versions"]
                                                    else conv_layers_over_5_dic["default"]["value"],
                   "use_maxpool_after_conv_layers_after_5th": use_maxpool_after_conv_layers_after_5th_dic[i] if i in use_maxpool_after_conv_layers_after_5th_dic else [],
                   "version": i}
    model = t_6_v1.trainModel( **params_dict )
    model_file_name = r"J:\Visible_models\model_6classes_v" + str(i) + ".h5"
    model.save(model_file_name)
    del model

#model = load_model (model_file_name)

#exec('Model_6class.conf_matrix_6classes.py')