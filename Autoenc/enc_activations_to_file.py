# Write activations of auto-encoder's encoded layer to file
from tensorflow.keras.backend import function
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import pickle

# Where to put activations files
autoenc_encoded_activations_folder = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc"
train_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"train_activations_enc.obj"])
val_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"val_activations_enc.obj"])
test_activations_filename = "\\".join ([autoenc_encoded_activations_folder,"test_activations_enc.obj"])

# Data files (source)
train_folder = "C:\\TrainAndVal_6classes\\Train"
val_folder = "C:\\TrainAndVal_6classes\\Val"
test_folder = "C:\\TrainAndVal_6classes\\Test"

# Load model
model_file_path = "J:\\Visible_models\\Autoenc\\model_autoenc_v2.h5"
model = load_model( model_file_path )

# How big is encoded layer (flattened)
enc_size = model.get_layer('flatten').output_shape[1]
# How big is image to resize to (exclude channels)
target_shape = model.layers[0].input_shape[1:3]

# Autoencoder's encoded layer activations
def get_activations_Encoded(model, X):

    enc_layer = model.get_layer('flatten')
    func_activation = function([model.input], [enc_layer.output])
    output_activation = func_activation([X])[0]
    return output_activation

# Paveikslelio paruosimo f-ja
def prepareImage (filename, target_shape):
    # Load image ( RGB )
    img = Image.open(filename)
	# Resize to target
    img = img.resize ( target_shape )
    img_arr = np.asarray (img)
    # Subtract global dataset average to center pixels to 0
    img_arr = img_arr / 255.
    return img_arr


# gets activations of pre-last layer
def get_all_activations_encoded (folder, model):
    # init array of all activations; shape [m,n], m - #samples; n - #neurons in pre-last layer
    all_activations_enc = np.empty( (0, enc_size ) )
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
                img_activations_enc = get_activations_Encoded (model, imgs)

                # add last image activations to all activations
                all_activations_enc = np.vstack ( [ all_activations_enc, img_activations_enc])
                all_filenames.append (file_name)
                all_classes.append (class_dir)
    print ("Shape all_activations_preLast:", all_activations_enc.shape)

    return (all_filenames, all_classes, all_activations_enc)





# Create a matrix of pre-last layer activations of train set, or load from file
print ("Preparing activations of train set's pre-last layer...")

if os.path.exists(train_activations_filename):
    #train_result = pickle.load( open(train_activations_filename, 'rb') )
    print ("File " + train_activations_filename + " already exists. Delete if needed")
else:
    train_result = get_all_activations_encoded (train_folder, model)
    pickle.dump(train_result, open(train_activations_filename, 'wb'))
    print ("Prepared train activations")



# Create a matrix of pre-last layer activations of val set, or load from file
print ("Preparing activations of val set's pre-last layer...")

if os.path.exists(val_activations_filename):
    #val_result = pickle.load( open(val_activations_filename, 'rb') )
    print ("File " + val_activations_filename + " already exists. Delete if needed")
else:
    val_result = get_all_activations_encoded (val_folder, model)
    pickle.dump(val_result, open(val_activations_filename, 'wb'))
    print ("Prepared val activations")


# Create a matrix of pre-last layer activations of test set, or load from file
print ("Preparing activations of test set's pre-last layer...")

if os.path.exists(test_activations_filename):
    #test_result = pickle.load( open(test_activations_filename, 'rb') )
    print ("File " + test_activations_filename + " already exists. Delete if needed")
else:
    test_result = get_all_activations_encoded (test_folder, model)
    pickle.dump(test_result, open(test_activations_filename, 'wb'))
    print ("Prepared test activations")