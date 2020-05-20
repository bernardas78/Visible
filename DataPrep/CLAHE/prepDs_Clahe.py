# Prepares dataset by CLAHE (contrast limited adaptive histogram equalization):
#
#   Src: "D:\\Visible_Data\\2.Cropped"
#   Dest:
#       "D:\\Visible_Data\\2b.Clahe"

from PIL import Image
import os
import cv2 as cv
import numpy as np

# Src
cropped_folder = "D:\\Visible_Data\\2.Cropped"
# Dest
dest_folder_clahe = "D:\\Visible_Data\\2b.Clahe"

# counter
img_cnt = 0

# 20200601.IntensityRemovalCLAHE.docx - parameter selection criteria
clahe = cv.createCLAHE(clipLimit=5, tileGridSize=(4,4))

for d_subcategory in os.listdir( cropped_folder ):

    # Create subcat dirs if needed for each subcat
    subcat_dir_clahe = os.path.join (dest_folder_clahe,d_subcategory)
    if not os.path.exists(subcat_dir_clahe):
        os.makedirs(subcat_dir_clahe)

    for filename in os.listdir( os.path.join(cropped_folder, d_subcategory) ):

        # Open each image
        img = Image.open ( os.path.join (cropped_folder, d_subcategory, filename) )
        img_arr = np.asarray (img)

        # Convert image to HSV
        img_hsv_arr = cv.cvtColor(img_arr, cv.COLOR_RGB2HSV)

        # Apply Clahe
        img_hsv_arr[:,:,2] = clahe.apply(img_hsv_arr[:, :, 2])  # replacing "V" channel of HSV


        # Convert subtracted_mean_intensity  to RGB
        img_rgb_arr = cv.cvtColor (img_hsv_arr, cv.COLOR_HSV2RGB)

        # Save clahe image
        clahe_img_rgb = Image.fromarray( img_rgb_arr )
        dest_full_filename = os.path.join (dest_folder_clahe,d_subcategory,filename)
        clahe_img_rgb.save( dest_full_filename )

        # Boring to wait
        if img_cnt%100 == 0:
            print ("Processed {} files".format(img_cnt))
        img_cnt += 1
