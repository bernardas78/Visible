# Create confusion matrix:

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import math
import os

# entropies file; used to load least entropy model for a given subcategory
entropies_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\entropies.csv"

# model file to load
model_file_pattern = r"J:\ClassMixture_Models\model_v{}.h5"

# resulting confusion matrix
conf_mat_file_name = os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\conf_mat.png"

# test files location
test_6class_folder = r"C:\TrainAndVal\Test_6class"

subcategories = ['1', '2', '3', '4', 'm', 'ma']

category_names = ["Invisible", "Visible"]

# vectors for construction of confusion matrix
visible_true = np.empty(0,dtype=np.int)
visible_pred = np.empty(0,dtype=np.int)

df_entropies = pd.read_csv(entropies_file_name)
#best_entropy_model_versions = {
#    "1": 0,
#    #"2": 0,
#    #"3": 0,
#    "4": 0,
#    #"m": 0,
#    "ma": 0
#}

# Data gen (common for all models)
image_datagen = ImageDataGenerator(rescale=1. / 255)
# print("subcat_test_folder:{}".format(test_folder))
test_image_generator = image_datagen.flow_from_directory(
    directory=test_6class_folder,
    shuffle=False)

for subcat in subcategories:

    #model_version = np.argmin(df_entropies[subcat])
    #print ("Best model for subcat {} is v{}".format(subcat, model_version))

    # Always use the model of least entropies weighted mean
    model_version = np.argmin(df_entropies["Mean_Weighted"])

    # Load and evaluate model on validation set
    model = load_model (model_file_pattern.format ( model_version ) )

    #print("Got test_image_generator")
    #print("len(test_image_generator):{}".format(len(test_image_generator)))
    predictions = model.predict_generator(test_image_generator, len(test_image_generator))

    # pred_classes has values 0 (Invisible) and 1 (Visible)
    pred_classes = np.argmax(predictions, axis=1)
    # print ("TOTAL PREDS VISIBLE, ALL: {},{}".format(np.sum(pred_classes), len(pred_classes)))

    # filter a single subcategory data
    subcat_pred_class_indices = np.where(test_image_generator.classes == test_image_generator.class_indices[subcat])[0]
    subcat_pred_classes = pred_classes[subcat_pred_class_indices]

    # Decide which category (Invis/Vis) a subcat should belong to (based on which has higher prob)
    subcat_frequentist_prob_visible = np.sum(subcat_pred_classes) / len(subcat_pred_classes)
    subcats_category_index = int ( round ( subcat_frequentist_prob_visible ) )
    print ("Subcat {} treated as {}, prob_visible:{}".format(subcat, category_names[subcats_category_index], subcat_frequentist_prob_visible ) )
    print ("#preds visible {} / {}".format(np.sum(subcat_pred_classes), len(subcat_pred_classes)))

    entropy_for_sanity_check = -subcat_frequentist_prob_visible * math.log(subcat_frequentist_prob_visible, 2) \
                               - (1 - subcat_frequentist_prob_visible) * math.log(1 - subcat_frequentist_prob_visible, 2)
    print ("Sanity check: entropies should match: {}, {}".format(np.min(df_entropies[subcat]), entropy_for_sanity_check) )

    # Add to true, pred vectors, later used to create confusion matrix
    #print ( "subcat_pred_classes.shape:{}, len(subcat_pred_classes): {}, type(subcat_pred_classes): {}".format(subcat_pred_classes.shape, len(subcat_pred_classes), type(subcat_pred_classes)) )
    visible_true = np.append( visible_true, np.repeat( subcats_category_index, len(subcat_pred_classes) ) )
    visible_pred = np.append( visible_pred, subcat_pred_classes )


print ("len(visible_true):{}, len(visible_pred):{}".format(len(visible_true), len(visible_pred)))

# Switch Invisible with Visible (display Visible top-left since it's a positive class)
conf_mat = confusion_matrix(y_true=(1-visible_true),y_pred=(1-visible_pred))

ax = sns.heatmap( np.round (conf_mat/np.sum(conf_mat)*100, decimals=1), annot=True, fmt='.1f', cbar=False )
for t in ax.texts: t.set_text(t.get_text() + " %")
ax.set_xticklabels( list (reversed(category_names)), horizontalalignment='right' ) # switch category names, like we switched data
ax.set_yticklabels( list (reversed( category_names)), horizontalalignment='right' )
ax.set_xlabel("PREDICTED", weight="bold")
ax.set_ylabel("ACTUAL", weight="bold")
plt.savefig (conf_mat_file_name)
plt.close()
