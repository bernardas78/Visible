# Boxplots of weights of each layer in a given model

from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
import os

#version = "v5"
#versions = [1,2,3,4,5,6,7,8,9,10,11,12,13] # weights.md
#versions = [13,14,15,16,17,18,19] # bn_impact_on_weights.md
#versions = [13,20,21] # dropout_impact_on_weights.md
#versions = [13,22,23,24] # l2_impact_on_weights.md
versions = [26,27,28,29,30,31] # cnn456_dense23_impact_on_weights.md

def showWeightsOfModel (version):
    model_file_name = r"J:\Visible_models\model_6classes_" + version + ".h5"
    weights_chart_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\Weights\model_6classes_" + version + "_weights.png"

    #data_dir_6classes_val = r"C:\TrainAndVal_6classes\Val"
    data_dir_6classes_val = r"D:\Visible_Data\3.SplitTrainVal\Val"

    # Load and evaluate model on validation set
    model = load_model(model_file_name)

    # Initialize a list of all layers weights
    all_layers_weights = []
    all_layer_names = []

    for layer in model.layers:
        # CNN and Dense layer is a list of 2 numpy arrays [ W, b ]
        layer_weights = layer.get_weights()
        if isinstance( layer_weights, list ) and len(layer_weights)>0:
            w_flat = np.ravel ( layer_weights[0] )
            b_flat = np.ravel ( layer_weights[1] )
            print (layer.name, len(w_flat), len(b_flat) )

            short_layer_name = layer.name.replace('batch_normalization','bn') # Replace long name with shorter for graphs to align
            all_layer_names.append (short_layer_name + "\n" + str(len(w_flat)) )
            all_layers_weights.append(w_flat)

            #all_layer_names.append (layer.name + " b")
            #all_layers_weights.append(b_flat)

    fig, ax = plt.subplots()
    ax.set_title('Weights of ' + model_file_name.split("\\")[-1])
    ax.boxplot(all_layers_weights, whis=[5,95])
    plt.xticks(np.arange(len(all_layer_names))+1, all_layer_names, rotation=90)
    plt.ylim(-0.6, 1.5)
    fig.tight_layout()  # fits rotated xticks inside image

    #plt.show()
    plt.savefig(weights_chart_file_name)
    plt.close()

for version in versions:
    showWeightsOfModel( "v" + str(version) )