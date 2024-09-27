# Returns cross entropy per subcategory given a trained model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import math
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ClassMixture import conf_mat

# model to load from
model_path_pattern = r"J:\ClassMixture_Models\model_v"

# resulting entropies file
entropy_filename = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\entropies.csv"

# resulting cross entropy and other metrics file
eval_metrics_filename = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\eval_metrics.csv"

# test files location
test_6class_folder = r"C:\TrainAndVal\Test_6class"

# conf mat file pattern
conf_mat_file_pattern = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\Conf_Mat_Eval\\conf_mat_v{}_grouping{}{}.png"

subcategories = ['1', '2', '3', '4', 'm', 'ma']

# experiment configurations (use for informative model names)
experiments_filename = "experiments_class_mixture.csv"

# if need to start in the middle due to memory overflow
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
    test_image_generator = image_datagen.flow_from_directory (
        directory= test_6class_folder,
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

    # for each subcategory, evaluate entropy
    for subcat in subcategories:

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

    #####################################################
    # EVALUATE CROSS ENTROPY BY ASSIGNING INTERIM SUB_CATS (,Q2,Q3,,m,ma) TO EITHER VIS OR INVIS
    #       VISIBILITY IS A DIRECTED GRAPH: Q1 -> Q2 -> Q3 -> Q4
    #                                       |--> ma --> m -|
    #####################################################
    for subcat_grouping_id in range(9):
        Subcats_Invisible_eval = [True, subcat_grouping_id%3>=1, subcat_grouping_id%3>=2, False, subcat_grouping_id/3>=2, subcat_grouping_id/3>=1 ]
        #print (Invisible_subcats_eval)

        # Given this subcat grouping, assign each actual (0-5) to actuals_grouped (0-Invis, 1-Vis)
        actuals_grouped = np.array([0 if Subcats_Invisible_eval[actual] else 1 for actual in actuals])

        # Cross entropy
        cross_entropies = [-(1-single_actual_grouped) * math.log(single_prediction[0], 2) \
                           -(single_actual_grouped) * math.log(single_prediction[1] ,2) \
                           for (single_actual_grouped,single_prediction) in zip(actuals_grouped,predictions) ]
        mean_cross_entropy = np.mean(cross_entropies)

        # General classification metrics
        acc = accuracy_score(actuals_grouped, pred_classes),
        prec = precision_score(actuals_grouped, pred_classes),
        recall = recall_score(actuals_grouped, pred_classes),
        f1 = f1_score(actuals_grouped, pred_classes)

        Invisible_subcats_eval = [subcategory for (subcat_invisible,subcategory) in zip(Subcats_Invisible_eval,subcategories) if subcat_invisible]
        Visible_subcats_eval = [subcategory for (subcat_invisible,subcategory) in zip(Subcats_Invisible_eval,subcategories) if not subcat_invisible]
        eval_grouping = "{}/{}".format(Visible_subcats_eval, Invisible_subcats_eval)
        df_metrics = pd.DataFrame(columns=["version", "eval_grouping", "acc", "prec", "recall", "f1", "mean_cross_entropy"],
                                  data=[np.hstack([version, eval_grouping, acc, prec, recall, f1, mean_cross_entropy])])

        df_metrics.to_csv(eval_metrics_filename, header=None, index=None, mode="a")
        print("Test accuracy: {}, precision: {}, recall: {}, f1: {}".format(acc, prec, recall, f1))

        # Confusion matrix
        conf_mat_filename = conf_mat_file_pattern.format(version,Visible_subcats_eval,Invisible_subcats_eval)
        conf_mat.twoclass_conf_mat_to_file(y_true=actuals_grouped, y_pred=pred_classes, conf_mat_filename=conf_mat_filename)

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
    df_entropies = pd.DataFrame( columns=["Model (Vis/Invis)",*subcategories,"Mean","Mean_Weighted"])
    df_entropies.to_csv(entropy_filename, header=True, index=None, mode='w')

    df_clsf_metrics = pd.DataFrame( columns=["version", "eval_grouping", "acc", "prec", "recall", "f1", "mean_cross_entropy"])
    df_clsf_metrics.to_csv(eval_metrics_filename, header=True, index=None, mode='w')

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