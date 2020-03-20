# Random neuron output of each layer in a given model, given picture

from ErrorAnalysis import get_activations
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
import os

# Model versions to produce feature maps for
#versions = [1,2,3,4,5,6,7,8,9,10,11,12,13]
#versions = [13,14,15,16,17,18,19] #bn_impact_on_feature_maps.md
#versions = [13,20,21] # dropout_impact_on_feature_maps.md
#versions = [13,22,23,24] # l2_impact_on_feature_maps.md
versions = [14,15,16,17,18,19,20,21,22,23,24] # l2_impact_on_feature_maps.md

picture_filenames = [
    r"D:\Visible_Data\3.SplitTrainVal\Val\1\000000005315_6_20190905145918346.jpg",
    r"D:\Visible_Data\3.SplitTrainVal\Val\4\00001157441_5_20191003184503612.jpg"
]

# Where to create feature maps?
activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Inside_cnn\\"

target_shape = (224,224)
grid_size = [8,8]

def show_inside_cnn_layers (version, model, picture_filename):

    img = Image.open(picture_filename).resize(target_shape)
    imgs_arr = np.asarray(img)[np.newaxis] / 255.

    # Strip path and extension
    picture_filename_pathless = picture_filename.split("\\")[-1].split(".")[0]

    # Save original image
    plt.imsave (activations_folder + str(version) + "\\" + picture_filename_pathless + "_0_original.jpg", (imgs_arr[0,:,:,:]*255).astype(np.uint8) )

    (layernames_list, activation_list) = get_activations.get_activations(model, imgs_arr)

    for layer_id in range(len(layernames_list)):

        # Get layer's activations. Last [0] indicates sample id
        layer_activations = activation_list[layer_id][0]


        if not 'conv' in layernames_list[layer_id] and not 'pool' in layernames_list[layer_id]:
            continue

        # Pick random 64 layer's activations
        rand_activations = np.random.randint (low=0, high=layer_activations.shape[2], size=64)

        # Produce a single image
        f, axarr = plt.subplots(grid_size[0], grid_size[1],
                                gridspec_kw={'wspace': 0.0, 'hspace': 0.05, 'right': 1.0, 'left': 0.0, 'top': 1.0,
                                             'bottom': 0.0})
        f.set_size_inches(6, 6)

        # Up to 64 subplots
        for subplot_id in range(len(rand_activations)):  # shape[3] = #layers of activation
            activation_id = rand_activations[subplot_id]

            row = math.floor(subplot_id / grid_size[1])
            col = subplot_id % grid_size[1]

            # Show actual activations
            axarr[row, col].imshow( layer_activations[:, :, activation_id])

            # no labels or markings on axes
            _ = axarr[row, col].set_xticklabels([])
            _ = axarr[row, col].set_yticklabels([])
            _ = axarr[row, col].set_xticks([])
            _ = axarr[row, col].set_yticks([])

        # Activations file name: <original>_<layer>.png
        act_file_name = activations_folder + str(version) + "\\" + picture_filename_pathless + "_" + str(layer_id+1) + "_" + layernames_list[layer_id] + ".png"

        # plt.show()
        plt.savefig(act_file_name)
        plt.close()


for version in versions:

    # Load and evaluate model on validation set
    model_file_name = r"J:\Visible_models\model_6classes_v" + str(version) + ".h5"
    model = load_model(model_file_name)

    # Create a folder for single image's interim layers pics
    activation_image_folder = activations_folder + str(version)
    if not os.path.exists(activation_image_folder):
        os.makedirs(activation_image_folder)

    # Evaluate each picture
    for picture_filename in picture_filenames:
        show_inside_cnn_layers(version, model, picture_filename)

    del model