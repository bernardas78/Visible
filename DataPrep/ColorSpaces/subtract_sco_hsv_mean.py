# Prepares dataset by subtracting SCO's HSV mean pictures "V" channel only
#   uses: pre-calculated means for each SCO in hsv_means_folder (made by by calculate_sco_hsv_means.py)

#   Src: "D:\\Visible_Data\\2.Cropped_BySCO"
#   Dest:
#       "D:\\Visible_Data\\2a.Subtracted_SCO_HSV_MeanIntensity"

from PIL import Image
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Src
cropped_folder = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
dest_folder_minus_mean_insensity = "D:\\Visible_Data\\2a.Subtracted_SCO_HSV_MeanIntensity"
# Means folder
hsv_means_folder = "D:\\Visible_Data\\ColorSpaces\\HSV_Means"

# counter
img_cnt = 0

for d_sco in os.listdir(cropped_folder):
    #print ("Here is a new SCO: {}".format(d_sco))

    # Load SCO's mean image
    img_sco_mean = Image.open(os.path.join(hsv_means_folder, d_sco + "_Mean.jpg"))
    img_sco_mean_arr = np.asarray(img_sco_mean)
    # Convert to HSV
    img_sco_hsv_mean_arr = cv.cvtColor (img_sco_mean_arr, cv.COLOR_RGB2HSV).astype(np.float) # converting to float because uint8(1)-uint8(2)=255

    for d_subcategory in os.listdir( os.path.join(cropped_folder, d_sco) ):

        # Create subcat dirs if needed
        subcat_dir_minus_intensity_path = os.path.join (dest_folder_minus_mean_insensity,d_subcategory)
        if not os.path.exists(subcat_dir_minus_intensity_path):
            os.makedirs(subcat_dir_minus_intensity_path)

        for filename in os.listdir( os.path.join(cropped_folder, d_sco, d_subcategory) ):

            # Open each image
            img = Image.open ( os.path.join (cropped_folder,d_sco, d_subcategory, filename) )
            img_arr = np.asarray (img)

            # Convert image to HSV
            img_hsv_arr = cv.cvtColor(img_arr, cv.COLOR_RGB2HSV).astype(np.float) # converting to float because uint8(1)-uint8(2)=255

            # Subtract HSV "V" channel mean
            subtracted_mean_intensity_img_hsv_arr = np.copy(img_hsv_arr)
            subtracted_mean_intensity_img_hsv_arr[:,:,2] -= img_sco_hsv_mean_arr[:,:,2]

            # Scale subtracted_mean_intensity between 0 and 255 (only "V" channel)
            (minus_mean_intensity_min, minus_mean_intensity_max) = np.min(subtracted_mean_intensity_img_hsv_arr[:,:,2]), np.max(subtracted_mean_intensity_img_hsv_arr[:,:,2])
            subtracted_mean_intensity_img_hsv_arr[:,:,2] = (subtracted_mean_intensity_img_hsv_arr[:,:,2] - minus_mean_intensity_min) / (minus_mean_intensity_max-minus_mean_intensity_min)*255

            # Convert subtracted_mean_intensity  to RGB
            subtracted_mean_intensity_img_rgb_arr = cv.cvtColor (subtracted_mean_intensity_img_hsv_arr.astype(np.uint8), cv.COLOR_HSV2RGB)

            # Save subtracted_mean_intensity
            subtracted_mean_intensity_img_rgb = Image.fromarray( subtracted_mean_intensity_img_rgb_arr )
            dest_full_filename = os.path.join (dest_folder_minus_mean_insensity,d_subcategory,filename)
            subtracted_mean_intensity_img_rgb.save( dest_full_filename )

            # Boring to wait
            if img_cnt%100 == 0:
                print ("Processed {} files".format(img_cnt))
            img_cnt += 1



#print (np.min(subtracted_mean_intensity_img_hsv_arr), np.max(subtracted_mean_intensity_img_hsv_arr))
#to_show = np.copy(img_hsv_arr).astype(np.uint8)
#to_show = cv.cvtColor(to_show, cv.COLOR_HSV2RGB)
#plt.imshow(to_show)
#plt.show()
