# Create classifier using SVM
#      on pre-last layer's activations of trained NN classifier
#           activations written to file by ErrorAnalysis\mispred_activations_preLast_to_file_for_knn.py

import pickle
import os
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

subcategories = ["1","2","3","4","m","ma"]

mispred_knn_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Misclsf_Knn"

#train_folder = "D:\\Visible_Data\\3.SplitTrainValTest\\Train"
#val_folder = "D:\\Visible_Data\\3.SplitTrainValTest\\Val"
#test_folder = "D:\\Visible_Data\\3.SplitTrainValTest\\Test"
train_folder = "C:\\TrainAndVal_6classes\\Train"
val_folder = "C:\\TrainAndVal_6classes\\Val"
test_folder = "C:\\TrainAndVal_6classes\\Test"

#   Intermediate file for train, val activations
train_activations_filename = "\\".join ([mispred_knn_folder,"train_activations_preLast.obj"])
val_activations_filename = "\\".join ([mispred_knn_folder,"val_activations_preLast.obj"])
test_activations_filename = "\\".join ([mispred_knn_folder,"test_activations_preLast.obj"])

# Load activation of train and test set
train_result = pickle.load( open(train_activations_filename, 'rb') )
val_result = pickle.load( open(val_activations_filename, 'rb') )
test_result = pickle.load( open(test_activations_filename, 'rb') )
print ("Loaded activations files")

# unpack train activations, file names, classes
(train_filenames, train_classes, train_pred_scores, train_activations_preLast) = train_result
(val_filenames, val_classes, val_pred_scores, val_activations_preLast) = val_result
(test_filenames, test_classes, test_pred_scores, test_activations_preLast) = test_result
print ("Unpacked activations: ")
print ("  train_filenames {0}, train_classes {1}, train_activations_preLast {2} ".format(len(train_filenames), len(train_classes), train_activations_preLast.shape))
print ("  val_filenames {0}, val_classes {1}, val_pred_scores {2}, val_activations_preLast {3} ".format(len(val_filenames), len(val_classes), val_pred_scores.shape, val_activations_preLast.shape))
print ("  test_filenames {0}, test_classes {1}, test_pred_scores {2}, test_activations_preLast {3} ".format(len(test_filenames), len(test_classes), test_pred_scores.shape, test_activations_preLast.shape))

# Train 6 class classifier
print ("Training 6 class SVC...")
#svm_clsf = LinearSVC(max_iter=1000000)
svm_clsf = SVC(probability=True)
svm_clsf.fit(train_activations_preLast, train_classes)
print ("Done training 6 class SVC")

# Evaluate 6 class classifier on train set, val set - (this is 1vsRest classifier)
#train_pred_scores = svm_clsf.predict_proba(train_activations_preLast)
print ("Evaluation 6 class on train, val sets")
train_pred = svm_clsf.predict(train_activations_preLast)
val_pred = svm_clsf.predict(val_activations_preLast)
test_pred = svm_clsf.predict(test_activations_preLast)
#train_conf_mat = confusion_matrix(train_classes, train_pred)
print ( '6 class Train accuracy {}; Val accuracy {}; Test accuracy {}'.format(
    accuracy_score(train_classes, train_pred),
    accuracy_score(val_classes, val_pred),
    accuracy_score(test_classes, test_pred)
) )

# Calculate Visible clases
train_visible_classes = [train_class in ['3','4','m'] for train_class in train_classes ]
val_visible_classes = [val_class in ['3','4','m'] for val_class in val_classes ]
test_visible_classes = [test_class in ['3','4','m'] for test_class in test_classes ]

# Train 2 class classifier: 1,2,ma vs 3,4,m
print ("Training 2 class SVC...")
#svm_clsf = LinearSVC(max_iter=1000000)
svm_visible_clsf = SVC(probability=True)
svm_visible_clsf.fit(train_activations_preLast, train_visible_classes)
print ("Done training 6 class SVC")

# Evaluate 6 class classifier on train set, val set - (this is 1vsRest classifier)
#train_pred_scores = svm_clsf.predict_proba(train_activations_preLast)
print ("Evaluation 2 class on train, val sets")
train_visible_pred = svm_visible_clsf.predict(train_activations_preLast)
val_visible_pred = svm_visible_clsf.predict(val_activations_preLast)
test_visible_pred = svm_visible_clsf.predict(test_activations_preLast)
#train_conf_mat = confusion_matrix(train_classes, train_pred)
print ( '2 class Train accuracy {}; Val accuracy {}; Test accuracy {}'.format(
    accuracy_score(train_visible_classes, train_visible_pred),
    accuracy_score(val_visible_classes, val_visible_pred),
    accuracy_score(test_visible_classes, test_visible_pred)
) )
