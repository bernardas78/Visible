from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from matplotlib import pyplot as plt
import math
import os


version = 30
model_file_name = r"J:\Visible_models\model_6classes_v" + str(version) + ".h5"
hypot_metrics_file_name = os.environ['GDRIVE'] + "\PhD_Data\Visible_ErrorAnalysis\Hypot_2_3_classes\model_6classes_v" + str(version) + "_hypot_metricst.png"

#data_dir_6classes_val = r"C:\TrainAndVal_6classes\Val"
data_dir_6classes_val = r"D:\Visible_Data\3.SplitTrainVal\Val"

# Load and evaluate model on validation set
model = load_model(model_file_name)

#target_size = 224  # for v<=26
target_size = 256
batch_size = 64

class_names = os.listdir(data_dir_6classes_val)

dataGen = ImageDataGenerator(rescale=1. / 255)

val_iterator = dataGen.flow_from_directory(directory=data_dir_6classes_val, target_size=(target_size, target_size),
                                           batch_size=batch_size, shuffle=False, class_mode='categorical')

Y_pred = model.predict_generator(val_iterator, len(val_iterator))
y_pred = np.argmax(Y_pred, axis=1)

# Initialize rectangular to store metrics for each combination and array of class names
metrics_mat = np.empty((0,4))
metrics_names = ["Accuracy","Precision","Recall","F1"]
config_names = []


static_visible_class_names = ["3","4"]
static_invisible_class_names = ["1","ma"]

all_class_indices = np.arange( len(class_names) )

# Static class indices
static_invisible_class_binary_vector = [ class_names[i] in static_invisible_class_names for i in range(len(class_names)) ]
static_visible_class_binary_vector = [ class_names[i] in static_visible_class_names for i in range(len(class_names)) ]
static_invisible_indices = all_class_indices[static_invisible_class_binary_vector]
static_visible_indices = all_class_indices[static_visible_class_binary_vector]

# classes that can be in either Visible or Not (all except static_visible and static_invisible)
dynamic_class_indices = all_class_indices [ ~np.bitwise_or (static_invisible_class_binary_vector , static_visible_class_binary_vector)  ]

# 2^4 combinations: move 2,3,m,ma subs to either category
for i in range( round(math.pow(2, len(dynamic_class_indices)))):

    # Generate current combination of subcats into visible cats
    dynamic_visible_binary_vector = [ math.floor ( round(i%math.pow(2,j+1)) / round(math.pow(2,j)))==1   for j in range(len(dynamic_class_indices)) ]
    dynamic_visible_indices = dynamic_class_indices [ dynamic_visible_binary_vector ]

    # Include static and dynamic visible
    visible_indices = np.concatenate ( (dynamic_visible_indices, static_visible_indices ))
    #print (visible_indices)

    y_pred_binary = np.isin ( y_pred, visible_indices )
    y_true_binary = np.isin(val_iterator.classes, visible_indices)

    (acc, prec, recall, f1) = (
        accuracy_score(y_true_binary, y_pred_binary),
        precision_score(y_true_binary, y_pred_binary),
        recall_score(y_true_binary, y_pred_binary),
        f1_score(y_true_binary, y_pred_binary) )
    #print (acc, prec, recall, f1)

    # Append Metrics matrix
    metrics_mat = np.vstack ( ( metrics_mat, np.round([acc, prec, recall, f1],3) ) )

    # Append configuration vector
    config_names.append( ",".join ([ class_names[visible_ind] for visible_ind in visible_indices ] ) )

max_in_each_column = np.max(metrics_mat, axis=0)

ax = sns.heatmap( metrics_mat, mask=max_in_each_column!=metrics_mat, annot_kws={"weight": "bold"}, annot=True, fmt='g', cbar=False )    #/np.sum(conf_mat)
ax = sns.heatmap( metrics_mat, mask=max_in_each_column==metrics_mat, fmt='g', annot=True, cbar=False )    #/np.sum(conf_mat)

#ax = sns.heatmap( metrics_mat, annot=True, fmt='g' )    #/np.sum(conf_mat)
ax.set_xticklabels( metrics_names, horizontalalignment='right' )
ax.set_yticklabels( config_names, rotation=0, horizontalalignment='right' )

plt.xlabel ("Metrics")
plt.ylabel ("Sub-categories of VISIBLE")

plt.savefig (hypot_metrics_file_name)
plt.close()
