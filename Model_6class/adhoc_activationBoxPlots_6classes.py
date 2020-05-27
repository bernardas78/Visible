# Boxplots of activations of each layer in a given model, given picture (used to show Batch Norm impact)

from ErrorAnalysis import get_activations
from tensorflow.keras.models import load_model
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

# Where to create activation box plots?
activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Activations\\"

# v13 - no batch normalization; v14-18 - added BN in various layers
#versions = [13,14,15,16,17,18,19] #bn_impact_on_activations.md
#versions = [13,20,21] # dropout_impact_on_activations.md
#versions = [13,22,23,24] # l2_impact_on_activations.md
versions = [26,27,28,29,30,31] # cnn456_dense23_impact_on_activations.md #target_shape = (224,) for v26, (256,) since

picture_filenames = [
    #r"D:\Visible_Data\3.SplitTrainVal\Val\1\000000005315_6_20190905145918346.jpg",
    #r"D:\Visible_Data\3.SplitTrainVal\Val\4\00001157441_5_20191003184503612.jpg"
    r"D:\Visible_Data\3.SplitTrainVal\Train\1\000000005315_6_20190905145918346.jpg",
    r"D:\Visible_Data\3.SplitTrainVal\Train\4\00001157441_5_20191003184503612.jpg"
]

#target_shape = (224,224)
target_shape = (256,256)

def show_activationBoxPlots (version, model, picture_filename):

    img = Image.open(picture_filename).resize(target_shape)
    imgs_arr = np.asarray(img)[np.newaxis] / 255.

    # Strip path and extension
    picture_filename_pathless = picture_filename.split("\\")[-1].split(".")[0]

    activations_boxplot_file_name = activations_folder + str(version) + "\\" + picture_filename_pathless + "_activations.png"

    # Save original image
    #plt.imsave (activations_folder + str(version) + "\\" + picture_filename_pathless + "_0_original.jpg", (imgs_arr[0,:,:,:]*255).astype(np.uint8) )

    (layernames_list, activation_list) = get_activations.get_activations(model, imgs_arr)

    # Initialize a list of all layers weights
    all_layers_activations = []
    all_layer_names = []


    for layer_id in range(len(layernames_list)):
        layer_name = layernames_list[layer_id]

        # Skip activation, maxpool, flatten layers since they are all above 0
        if "activation" in layer_name or "max_pooling" in layer_name or "flatten" in layer_name:
            continue

        # get current layer activations, in 1D format
        layer_activations = activation_list[layer_id].ravel()

        all_layer_names.append(layer_name.replace('batch_normalization','bn') )
        all_layers_activations.append(layer_activations)

    fig, ax = plt.subplots()
    #ax.set_title('Activations of ' + picture_filename_pathless + ', ' + model_file_name.split("\\")[-1])
    ax.boxplot(all_layers_activations, whis=[5,95])
    plt.xticks(np.arange(len(all_layer_names))+1, all_layer_names, rotation=90)
    fig.tight_layout() # fits rotated xticks inside image
    plt.ylim(-10,10)

    #plt.show()
    plt.savefig(activations_boxplot_file_name)
    plt.close()


for version in versions:

    # Load and evaluate model on validation set
    model_file_name = r"J:\Visible_models\6class\model_6classes_v" + str(version) + ".h5"
    model = load_model(model_file_name)

    # Create a folder for single image's interim layers pics
    activation_image_folder = activations_folder + str(version)
    if not os.path.exists(activation_image_folder):
        os.makedirs(activation_image_folder)

    # Evaluate each picture
    for picture_filename in picture_filenames:
        show_activationBoxPlots(version, model, picture_filename)

    del model