# Displays original+encoded image side by side from validation set
#

from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np

model_file_path = "J:\\Visible_models\\Autoenc\\model_autoenc_v2.h5"
data_path = "C:\\TrainAndVal_6classes\\Val\\4"
autoenc_show_path = os.environ['GDRIVE'] + "\\PhD_Data\\Visible_ErrorAnalysis\\Autoenc"

cnt_files_to_display = 10

model = load_model( model_file_path )

for i,filename in enumerate(os.listdir(data_path)):

    image_filepath = os.path.join(data_path, filename)

    img_arr = np.asarray ( Image.open(image_filepath).resize((256,256)) ) /255
    imgs_arr = np.expand_dims (img_arr, axis=0) # need to pass array of images [?,height,width,channels]

    imgs_pred_arr = model.predict(imgs_arr)
    img_pred_arr = imgs_pred_arr[0,:,:,:] # back to [height,width,channels] from [?,height,width,channels]

    # 2 images side by side : original and predicted (endoded->decoded)
    img_both_arr = np.zeros((img_arr.shape[0], img_arr.shape[1] * 2, 3), dtype=np.uint8)
    img_both_arr[:, :img_arr.shape[1], :] = np.round(np.array(img_arr) * 255)
    img_both_arr[:, img_arr.shape[1]:, :] = np.round(img_pred_arr * 255).astype(np.uint8)

    img_both = Image.fromarray(img_both_arr)
    img_both.save( os.path.join(autoenc_show_path, filename) )

    if i>=cnt_files_to_display:
        break