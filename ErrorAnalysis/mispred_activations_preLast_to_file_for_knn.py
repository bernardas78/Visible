# For mis-predictions in validation set:
#   Create activations files for train set and validation set
#       Activations file contains: activations, prediction scores, file names, actual classes

import os
from tensorflow.keras.models import load_model
import numpy as np
from ErrorAnalysis import get_activations
import cv2 as cv
import pickle

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

model_filename = "j:\\Visible_models\\model_6classes_v44.h5"
print ("Loading model...")
model = load_model(model_filename)
print ("Loaded model")

target_shape = ( model.layers[0].input_shape[1], model.layers[0].input_shape[2] )

# Paveikslelio paruosimo f-ja
def prepareImage (filename, target_shape):
	# Load image ( BGR )
	img = cv.imread(filename)
	# Resize to target
	img = cv.resize ( img, target_shape )
	# Subtract global dataset average to center pixels to 0
	img = img / 255.
	return img

# gets activations of pre-last layer
def get_all_activations_preLast (folder, model):
    # init array of all activations; shape [m,n], m - #samples; n - #neurons in pre-last layer
    all_activations_preLast = np.empty( (0, model.layers[-2].output_shape[1] ) )
    all_pred_scores = np.empty( (0, model.layers[-1].output_shape[1] ) )
    all_filenames = []
    all_classes = []
    # collect activations of images

    i=0
    for _,class_dirs,_ in os.walk(folder):
        for class_dir in class_dirs:
            print("Current class: ",class_dir,folder)
            for file_name in os.listdir("\\".join([folder , class_dir]) ):
                i+=1
                if i%20==0:
                    print ("Processed {0} images ".format(i) )
                img_preped = prepareImage ( "\\".join ( [folder,class_dir,file_name] ), target_shape )
                imgs = np.stack ( [img_preped] ) #quick way to add a dimension
                img_activations_preLast = get_activations.get_activations_preLast (model, imgs)
                img_pred_scores = model.predict(imgs)

                # add last image activations to all activations
                all_activations_preLast = np.vstack ( [ all_activations_preLast, img_activations_preLast])
                all_pred_scores = np.vstack ( [ all_pred_scores, img_pred_scores ] )
                all_filenames.append (file_name)
                all_classes.append (class_dir)
    print ("Shape all_activations_preLast:", all_activations_preLast.shape)

    return (all_filenames, all_classes, all_pred_scores, all_activations_preLast)

# Create a matrix of pre-last layer activations of train set, or load from file
print ("Preparing activations of train set's pre-last layer...")

if os.path.exists(train_activations_filename):
    #train_result = pickle.load( open(train_activations_filename, 'rb') )
    print ("File " + train_activations_filename + " already exists. Delete if needed")
else:
    train_result = get_all_activations_preLast (train_folder, model)
    pickle.dump(train_result, open(train_activations_filename, 'wb'))
print ("Prepared train activations")

if os.path.exists(val_activations_filename):
    #val_result = pickle.load( open(val_activations_filename, 'rb') )
    print ("File " + val_activations_filename + " already exists. Delete if needed")
else:
    val_result = get_all_activations_preLast(val_folder, model)
    pickle.dump(val_result, open(val_activations_filename, 'wb'))
print ("Prepared val activations")

if os.path.exists(test_activations_filename):
    #test_result = pickle.load( open(test_activations_filename, 'rb') )
    print ("File " + test_activations_filename + " already exists. Delete if needed")
else:
    test_result = get_all_activations_preLast(test_folder, model)
    pickle.dump(test_result, open(test_activations_filename, 'wb'))
print ("Prepared test activations")


# unpack train activations, file names, classes
#(train_filenames, train_classes, train_pred_scores, train_activations_preLast) = train_result
#(val_filenames, val_classes, val_pred_scores, val_activations_preLast) = val_result
#(test_filenames, test_classes, test_pred_scores, test_activations_preLast) = test_result

