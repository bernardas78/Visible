# Calculate HSV means of dataset and save as images

#   Src: "D:\\Visible_Data\\2.Cropped", "D:\\Visible_Data\\2.Cropped_BySCO"
#   Dest: "D:\\Visible_Data\\ColorSpaces\\HSV_Means"

# Calculate overall mean
do_overall_subcat_mean = True

# Calculate mean by SCO
do_sco_mean = True

# Create a few sample images (how they look after subtracting means)
do_sample_images = True

from PIL import Image
import numpy as np
import os
import cv2 as cv

# Src
cropped_folder_by_sco = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
hsv_means_folder = "D:\\Visible_Data\\ColorSpaces\\HSV_Means"


# Init overall mean
img_hsv_mean_arr = np.zeros( (360,360,3), dtype=np.float )
img_cnt = 0

for d_sco in os.listdir(cropped_folder_by_sco):
    for d_subcat in os.listdir( os.path.join (cropped_folder_by_sco,d_sco  ) ):

        # Init sco mean
        img_sco_hsv_mean_arr = np.zeros((360, 360, 3), dtype=np.float)
        img_sco_cnt = 0

        for img_filename in os.listdir( os.path.join( cropped_folder_by_sco, d_sco, d_subcat ) ):

            full_img_filename = os.path.join( cropped_folder_by_sco, d_sco, d_subcat, img_filename )
            #print (full_img_filename)
            img_rgb = Image.open(full_img_filename)
            img_rgb_arr = np.asarray(img_rgb)

            # Convert image to HSV
            img_hsv_arr = cv.cvtColor(img_rgb_arr, cv.COLOR_RGB2HSV)

            # update overall mean image
            img_hsv_mean_arr = (img_hsv_mean_arr*img_cnt + img_hsv_arr ) / (img_cnt+1)
            img_cnt +=1

            # update sco mean image
            img_sco_hsv_mean_arr = (img_sco_hsv_mean_arr*img_sco_cnt + img_hsv_arr ) / (img_sco_cnt+1)
            img_sco_cnt +=1

            # It's boring to wait
            if img_cnt%100==0:
                print ("Processed {} images".format(img_cnt))

    if do_sco_mean:
        # Convert back to RGB
        img_sco_rgb_mean_arr = cv.cvtColor(img_sco_hsv_mean_arr.astype(np.uint8), cv.COLOR_HSV2RGB)
        # Save sco mean
        img_sco_mean = Image.fromarray(img_sco_rgb_mean_arr.astype(np.uint8))
        img_sco_mean.save(os.path.join(hsv_means_folder, d_sco + "_Mean.jpg"))

if do_overall_subcat_mean:
    # Convert back to RGB
    img_rgb_mean_arr = cv.cvtColor(img_hsv_mean_arr.astype(np.uint8), cv.COLOR_HSV2RGB)
    # Save overall mean
    img_mean = Image.fromarray(img_rgb_mean_arr.astype(np.uint8))
    img_mean.save( os.path.join ( hsv_means_folder, "Overall_Mean.jpg") )



