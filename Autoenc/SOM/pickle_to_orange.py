# Converts autoencoder encoded activations from pickle files (.obj) to Orange table files (.tab)
#   Src: (.obj) created by ..\enc_activations_to_file.py
#   Dest: Orange table files (.tab) - later used by Orange and create_som_clusters.py (Orange API) to create SOM clusters and classifier


import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import pickle
import os
import numpy as np

subcategories = ["1","2","3","4","m","ma"]


# Src: Autoencoder's encoded activations
autoenc_encoded_activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc"
train_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"train_activations_enc.obj"])
val_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"val_activations_enc.obj"])
test_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"test_activations_enc.obj"])

# Dest:
orange_encoded_activations_folder = autoenc_encoded_activations_folder
orange_train_activations_filename = "\\".join ([orange_encoded_activations_folder,"train_activations_enc.tab"])
orange_val_activations_filename = "\\".join ([orange_encoded_activations_folder,"val_activations_enc.tab"])
orange_test_activations_filename = "\\".join ([orange_encoded_activations_folder,"test_activations_enc.tab"])

# Load activation files
(train_filenames, train_classes, train_activations_enc) = pickle.load( open(train_activations_filename, 'rb') )
(val_filenames, val_classes, val_activations_enc) = pickle.load( open(val_activations_filename, 'rb') )
(test_filenames, test_classes, test_activations_enc) = pickle.load( open(test_activations_filename, 'rb') )
print ("Loaded activations files")
print ("  train_filenames {0}, train_classes {1}, train_activations_enc {2} ".format(len(train_filenames), len(train_classes), train_activations_enc.shape))
print ("  val_filenames {0}, val_classes {1}, val_activations_enc {2} ".format(len(val_filenames), len(val_classes), val_activations_enc.shape))
print ("  test_filenames {0}, test_classes {1}, test_activations_enc {2} ".format(len(test_filenames), len(test_classes), test_activations_enc.shape))

# convert to Orange table and save
#   Later used for:
#       a) Visual Orange SOM
#       b) create_autoenc_som_clusters.py - SOM clustering+classifying
domain = Domain(
            [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(train_activations_enc.shape[1])],
            DiscreteVariable.make(name="subcategory", values=subcategories) )
train_class_indices = np.asarray ( [subcategories.index(train_class) for train_class in train_classes] )
train_tab = Orange.data.Table.from_numpy( domain=domain, X=train_activations_enc, Y=train_class_indices)

val_class_indices = np.asarray ( [subcategories.index(val_class) for val_class in val_classes] )
val_tab = Orange.data.Table.from_numpy( domain=domain, X=val_activations_enc, Y=val_class_indices)

test_class_indices = np.asarray ( [subcategories.index(test_class) for test_class in test_classes] )
test_tab = Orange.data.Table.from_numpy( domain=domain, X=test_activations_enc, Y=test_class_indices)

# Save Orange table files
train_tab.save (orange_train_activations_filename)
val_tab.save (orange_val_activations_filename)
test_tab.save (orange_test_activations_filename)

print ("Saved Orange files (train, val, test)")

