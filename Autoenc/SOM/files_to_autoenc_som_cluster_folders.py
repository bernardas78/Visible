# assign each file from train/val/test to a SOM cluster
# place each cluster's files in separate folder

import os
import pickle
import time
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
from Orange.projection import som
import numpy as np
from shutil import copyfile

subcategories = ["1","2","3","4","m","ma"]

#Image location
scr_folder = "C:\\4.Augmented_4823"

# Dest folder
clustered_files_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc\\ClusteredFiles"

# Load SOM clustering model (trained by create_autoenc_som_clusters.py)
print ( "Loading model...")
filehandler = open("J:\\Visible_models\\SOM\\som_v1.h5", 'rb')
mysom = pickle.load ( filehandler)
filehandler.close()
print ( "Done Loading model")

# Src: Autoencoder's encoded activations
autoenc_encoded_activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc"
train_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"train_activations_enc.obj"])
val_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"val_activations_enc.obj"])
test_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"test_activations_enc.obj"])

# Load activation files
print ( "Loading activations...")
(train_filenames, train_classes, train_activations_enc) = pickle.load( open(train_activations_filename, 'rb') )
(val_filenames, val_classes, val_activations_enc) = pickle.load( open(val_activations_filename, 'rb') )
(test_filenames, test_classes, test_activations_enc) = pickle.load( open(test_activations_filename, 'rb') )
print ("Loaded activations files")
print ("  train_filenames {0}, train_classes {1}, train_activations_enc {2} ".format(len(train_filenames), len(train_classes), train_activations_enc.shape))
print ("  val_filenames {0}, val_classes {1}, val_activations_enc {2} ".format(len(val_filenames), len(val_classes), val_activations_enc.shape))
print ("  test_filenames {0}, test_classes {1}, test_activations_enc {2} ".format(len(test_filenames), len(test_classes), test_activations_enc.shape))

# convert activations to Orange table
#   Later used for:
#       a) Visual Orange SOM
#       b) create_autoenc_som_clusters.py - SOM clustering+classifying
domain = Domain(
            [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(train_activations_enc.shape[1])],
            DiscreteVariable.make(name="subcategory", values=subcategories) )

def processSingleSet ( activations_enc, classes, filenames, theSet):
    # Create Orange Tables (used for SOM prediction)
    class_indices = np.asarray([subcategories.index(theclass) for theclass in classes])
    tab = Orange.data.Table.from_numpy(domain=domain, X=activations_enc, Y=class_indices)
    print ("Created Orange table")

    # Predict winning SOM's neuron
    now=time.time()
    pred_winner_neurons = mysom.winners(tab.X)
    print ("Predicted winners in {} sec".format(time.time()-now))

    # Copy each sample to winning neuron's folder
    for sample_id in np.arange(tab.X.shape[0]):
        # get the sample's winner
        winner_x, winner_y = pred_winner_neurons[sample_id, :]

        # create folder for winner neuron, in not exists
        winner_folder = os.path.join(clustered_files_folder, str(winner_x) + "_" + str(winner_y))
        if not os.path.exists(winner_folder):
            os.makedirs(winner_folder)

        # Copy file; filename = <subcat_origfilename>
        src_filename = os.path.join ( scr_folder, theSet, classes[sample_id], filenames[sample_id] )
        newfilename = os.path.join(winner_folder, classes[sample_id]+"_"+filenames[sample_id])
        copyfile(src_filename, newfilename)

print ( "Proessing test...")
processSingleSet (test_activations_enc, test_classes, test_filenames, "Test")
print ( "Proessing train...")
processSingleSet (train_activations_enc, train_classes, train_filenames, "Train")
print ( "Proessing val...")
processSingleSet (val_activations_enc, val_classes, val_filenames, "Val")

