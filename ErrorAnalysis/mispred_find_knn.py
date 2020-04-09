# For mis-predictions in validation set:
#   Find K closest neighbours from train set, save them to folder for review
#       Need to manually delete folders from mispred_knn_folder before running this
#
#   Saves intermediate file for train activations - recrete if unsure

import os
import numpy as np
import pickle
from shutil import copyfile

mispred_knn_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Misclsf_Knn"

#train_folder = "D:\\Visible_Data\\3.SplitTrainValTest\\Train"
#val_folder = "D:\\Visible_Data\\3.SplitTrainValTest\\Val"
train_folder = "C:\\TrainAndVal_6classes\\Train"
val_folder = "C:\\TrainAndVal_6classes\\Val"

#   Intermediate file for train, val activations
train_activations_filename = "\\".join ([mispred_knn_folder,"train_activations_preLast.obj"])
val_activations_filename = "\\".join ([mispred_knn_folder,"val_activations_preLast.obj"])

subcategories = ["1","2","3","4","m","ma"]

# how many nearest neighbors?
K=5

# Load activation of train and test set
train_result = pickle.load( open(train_activations_filename, 'rb') )
val_result = pickle.load( open(val_activations_filename, 'rb') )
print ("Loaded activations files")

# unpack train activations, file names, classes
(train_filenames, train_classes, train_pred_scores, train_activations_preLast) = train_result
(val_filenames, val_classes, val_pred_scores, val_activations_preLast) = val_result
print ("Unpacked activations: ")
print ("  train_filenames {0}, train_classes {1}, train_activations_preLast {2} ".format(len(train_filenames), len(train_classes), train_activations_preLast.shape))
print ("  val_filenames {0}, val_classes {1}, val_pred_scores {2}, val_activations_preLast {3} ".format(len(val_filenames), len(val_classes), val_pred_scores.shape, val_activations_preLast.shape))

# find k nearest neighbors using pre-last layer's activations (L2 norm of differences)
def getKnn (sample_preLast, train_activations_preLast):
    # dist_allNeurons - shame [m,n], where m - #samples in train set; n - #neurons in preLast
    dist_allNeurons = (train_activations_preLast - sample_preLast)
    # l2_dist - distance from a concrete sample; shape (m,)
    l2_dist = np.linalg.norm (dist_allNeurons, axis=1)

    knn_ind = np.argsort(l2_dist)[:K]
    knn_dist = l2_dist [knn_ind]
    return (knn_ind, knn_dist)


# For each validation image:
#   If mispredicted:
#       Save original to  Misclsf_Knn\original_<actual>_<predicted>.jpg
#       Find K nearest neighbors (smallest L2 norms of [val_img - train_img] )
#       Save Knn to Misclsf_Knn\<valfile>\d<dist>_actual_predicted_<trainfile>.jpg
for val_sample_ind in range ( len(val_filenames) ):
    # Is mispredicted?
    sample_pred_class_ind = np.argmax(val_pred_scores[val_sample_ind,:])
    sample_pred_class = subcategories[sample_pred_class_ind]
    if sample_pred_class != val_classes[val_sample_ind]:
        #       create folder (same as filename w/o ext)
        sample_orig_filename = val_filenames[val_sample_ind]
        sample_orig_filename_no_ext = sample_orig_filename.split(".")[0]
        os.mkdir( "\\".join ( [mispred_knn_folder,sample_orig_filename_no_ext] ) )

        # Save original and each close neighbour to folder
        src_full_filename = "\\".join ( [val_folder, val_classes[val_sample_ind], sample_orig_filename] )
        # format of new filename: original_<actual>_<predicted>.jpg
        dest_full_filename = "\\".join ( [mispred_knn_folder, sample_orig_filename_no_ext, "original_" + val_classes[val_sample_ind] + "_" + sample_pred_class + ".jpg" ] )
        copyfile(src_full_filename, dest_full_filename)

        # Find closest K neighbours
        sample_preLast = val_activations_preLast[val_sample_ind,:]
        (knn_ind, knn_dist) = getKnn(sample_preLast, train_activations_preLast)
        for idx,train_neighbour_id in enumerate(knn_ind):
            # found neighcour's distance from sample and predicted class
            neigbour_dist = '{:04d}'.format( int(knn_dist[idx]*10) ) # lowest distance tend to be >1,<10. multiply by 10 to differentiate
            neigbour_pred_class_ind = np.argmax(train_pred_scores[train_neighbour_id,:])
            neigbour_pred_class = subcategories[neigbour_pred_class_ind]

            src_full_filename = "\\".join([train_folder, train_classes[train_neighbour_id], train_filenames[train_neighbour_id] ] )
            # format of new filename: d<dist_from_sample>_<actual>_<predicted>_origfilename
            dest_filename = "d" + neigbour_dist + "_" + train_classes[train_neighbour_id] + "_"+ neigbour_pred_class + "_" + train_filenames[train_neighbour_id]
            dest_full_filename = "\\".join([mispred_knn_folder, sample_orig_filename_no_ext, dest_filename ] )
            copyfile(src_full_filename, dest_full_filename)

        #if val_sample_ind>10:
        #    break
