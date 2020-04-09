from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np

model_filename = "j:\\Visible_models\\model_6classes_v30.h5"
print ("Loading model...")
model = load_model(model_filename)
print ("Loaded model")

knn_folder = "D:\\Google Drive\\PhD_Data\\Visible_ErrorAnalysis\\Misclsf_Knn\\Using_3.SplitTrainTest\\000000005314_5_20190906121344255"

good_file_name = 'd0030_1_1_800214003930_10_20190905082318398.jpg'
bad_file_name = 'original_1_3.jpg'

# Paveikslelio paruosimo f-ja
def prepareImage (filename, target_shape):
	# Load image ( BGR )
	img = cv.imread(filename)
	# Resize to target
	img = cv.resize ( img, target_shape )
	# Subtract global dataset average to center pixels to 0
	img = img / 255.
	return img

img_good = prepareImage ( "\\".join ( [knn_folder,good_file_name] ), (256,256) )
img_bad = prepareImage ( "\\".join ( [knn_folder,bad_file_name] ), (256,256) )

imgs = np.stack ( [img_good, img_bad] )

img_pred_scores = model.predict(imgs)
print ("Pred scores:",img_pred_scores)
# both output the same class "3"