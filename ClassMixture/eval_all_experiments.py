# Returns cross entropy per subcategory given a trained model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import math
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

# model to load from
model_path_pattern = r"J:\ClassMixture_Models\model_v"

# resulting entropies file
entropy_filename = r"J:\ClassMixture_Metrics\entropies.csv"

# test files location
test_folder = r"C:\TrainAndVal\Test"

subcategories = ['1', '2', '3', '4', 'm', 'ma']

# experiment configurations (use for informative model names)
experiments_filename = "experiments_class_mixture.csv"

# if need to start in the middle
start_model = 0

def evalSingleClassMixtureExperiment (version):

    # list of entropy values for each subcategory
    subcat_entropies = []

    # Load model
    model_file_name = model_path_pattern + str(version) + ".h5"
    model = load_model (model_file_name)

    # Get predictions (for all classes)
    image_datagen = ImageDataGenerator(rescale=1. / 255)
    #print("subcat_test_folder:{}".format(test_folder))
    test_image_generator = image_datagen.flow_from_directory(
        directory=test_folder,
        shuffle=False,
        classes=None,
        class_mode=None)
    #print("Got test_image_generator")
    #print("len(test_image_generator):{}".format(len(test_image_generator)))
    predictions = model.predict_generator(test_image_generator, len(test_image_generator))

    # pred_classes has values 0 (Invisible) and 1 (Visible)
    pred_classes = np.argmax(predictions, axis=1)

    # actuals has values 0-5 (subcategories)
    actuals = test_image_generator.classes
    #print("predictions.shape: {}, actuals.shape:{}, np.unique(actuals,return_counts=True):{}".format(predictions.shape, actuals.shape, np.unique(actuals,return_counts=True) ) )
    #print ("test_image_generator.class_indices:{}".format(test_image_generator.class_indices) )

    # for each subcategory, evaluate cross entropy
    for subcat in subcategories:

        # filter by subcat

        # subcat_pred_classes has values 0 (Invisible) and 1 (Visible) for a certain subcategory
        subcat_pred_class_indices = np.where (test_image_generator.classes == test_image_generator.class_indices[subcat] )[0]
        subcat_pred_classes = pred_classes[subcat_pred_class_indices]

        # Calc cross entropy
        #print ("subcat_pred_classes.shape: {}".format (subcat_pred_classes.shape))
        subcat_frequentist_prob_visible = np.sum(subcat_pred_classes) / len(subcat_pred_classes)
        #print ("np.sum(subcat_pred_classes):{}, len(subcat_pred_classes):{}".format(np.sum(subcat_pred_classes), len(subcat_pred_classes)))
        #print ("subcat_pred_classes[-10:]:{}".format(subcat_pred_classes[-10:]))
        #print ("subcat:{} subcat_frequentist_prob_visible:{}".format(subcat, subcat_frequentist_prob_visible))
        if subcat_frequentist_prob_visible>1e-5 and subcat_frequentist_prob_visible<(1-1e-5):
            subcat_entropy = -subcat_frequentist_prob_visible * math.log(subcat_frequentist_prob_visible, 2) \
                             -(1-subcat_frequentist_prob_visible) * math.log(1-subcat_frequentist_prob_visible, 2)
        else:
            subcat_entropy = 0.

        subcat_entropies.append(subcat_entropy)

    del model
    K.clear_session()

    # Mean entropy
    mean_entropy = np.mean(subcat_entropies)

    # Mean weighted entropy
    subcat_counts = np.unique(actuals,return_counts=True)[1]
    mean_weighted_entropy = np.sum(subcat_entropies * subcat_counts) / np.sum(subcat_counts)
    #print("len(subcat_entropies): {}".format(len(subcat_entropies)))

    subcat_entropies.append(mean_entropy)
    subcat_entropies.append(mean_weighted_entropy)

    return subcat_entropies

# otherwise, continuing since memory error
if start_model == 0:
    df_accs = pd.DataFrame( columns=["Model (Vis/Invis)",*subcategories,"Mean","Mean_Weighted"])
    df_accs.to_csv(entropy_filename, header=True, index=None, mode='w')

# read experiment configurations (all models must be pretrained)
for version,row in pd.read_csv(experiments_filename).iterrows():

    # start since memory error
    if version<start_model:
        continue

    # Format model name
    Invisible_subcats = [ subcat for subcat in subcategories if row[subcat] == 0 ]
    Visible_subcats = [ subcat for subcat in subcategories if row[subcat] == 1 ]
    model_name = "{}/{}".format(Visible_subcats, Invisible_subcats)

    subcat_entropies = evalSingleClassMixtureExperiment(version)

    print ("Model:{}, entropies:{}".format (model_name, subcat_entropies))
    df_entropies = pd.DataFrame( data=[np.hstack([ model_name, *subcat_entropies])])
    df_entropies.to_csv(entropy_filename, header=None, index=None, mode='a')