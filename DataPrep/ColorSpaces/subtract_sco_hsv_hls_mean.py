# Prepares dataset by subtracting SCO's:
#       HSV mean pictures "V" channel only
#       HLS mean pictures "L" channel only
#   uses: pre-calculated means for each SCO in hsv_means_folder, hls_means_folder (made by by calculate_sco_hsv_hls_means.py)

#   Src: "D:\\Visible_Data\\2.Cropped_BySCO"
#        "D:\\Visible_Data\\ColorSpaces\\HSV_Means", "D:\\Visible_Data\\ColorSpaces\\HLS_Means"
#   Dest:
#       "D:\\Visible_Data\\2a.Subtracted_SCO_HSV_MeanIntensity"
#       "D:\\Visible_Data\\2a.Subtracted_SCO_HLS_MeanIntensity"

from PIL import Image
import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Src
cropped_folder = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
dest_folder_minus_hsv_mean_insensity = "D:\\Visible_Data\\2a.Subtracted_SCO_HSV_MeanIntensity"
dest_folder_minus_hls_mean_insensity = "D:\\Visible_Data\\2a.Subtracted_SCO_HLS_MeanIntensity"
# Means folder
hsv_means_folder = "D:\\Visible_Data\\ColorSpaces\\HSV_Means"
hls_means_folder = "D:\\Visible_Data\\ColorSpaces\\HLS_Means"

# counter
img_cnt = 0

for d_sco in os.listdir(cropped_folder):
    #print ("Here is a new SCO: {}".format(d_sco))

    # Load SCO's mean images
    img_sco_hsv_mean = Image.open(os.path.join(hsv_means_folder, d_sco + "_Mean.jpg"))
    img_sco_hsv_mean_arr = np.asarray(img_sco_hsv_mean)
    img_sco_hls_mean = Image.open(os.path.join(hls_means_folder, d_sco + "_Mean.jpg"))
    img_sco_hls_mean_arr = np.asarray(img_sco_hls_mean)

    # Convert to HSV, HLS
    img_sco_hsv_mean_arr = cv.cvtColor (img_sco_hsv_mean_arr, cv.COLOR_RGB2HSV).astype(np.float) # converting to float because uint8(1)-uint8(2)=255
    img_sco_hls_mean_arr = cv.cvtColor (img_sco_hls_mean_arr, cv.COLOR_RGB2HLS).astype(np.float) # converting to float because uint8(1)-uint8(2)=255

    for d_subcategory in os.listdir( os.path.join(cropped_folder, d_sco) ):

        # Create subcat dirs if needed for HSV, HLS
        subcat_dir_minus_hsv_intensity_path = os.path.join (dest_folder_minus_hsv_mean_insensity,d_subcategory)
        if not os.path.exists(subcat_dir_minus_hsv_intensity_path):
            os.makedirs(subcat_dir_minus_hsv_intensity_path)
        subcat_dir_minus_hls_intensity_path = os.path.join (dest_folder_minus_hls_mean_insensity,d_subcategory)
        if not os.path.exists(subcat_dir_minus_hls_intensity_path):
            os.makedirs(subcat_dir_minus_hls_intensity_path)

        for filename in os.listdir( os.path.join(cropped_folder, d_sco, d_subcategory) ):

            # Open each image
            img = Image.open ( os.path.join (cropped_folder,d_sco, d_subcategory, filename) )
            img_arr = np.asarray (img)

            # Convert image to HSV, HLS
            img_hsv_arr = cv.cvtColor(img_arr, cv.COLOR_RGB2HSV).astype(np.float) # converting to float because uint8(1)-uint8(2)=255
            img_hls_arr = cv.cvtColor(img_arr, cv.COLOR_RGB2HLS).astype(np.float) # converting to float because uint8(1)-uint8(2)=255

            # Subtract HSV "V" channel mean
            subtracted_mean_intensity_img_hsv_arr = np.copy(img_hsv_arr)
            subtracted_mean_intensity_img_hsv_arr[:,:,2] -= img_sco_hsv_mean_arr[:,:,2]

            # Subtract HLS "L" channel mean
            subtracted_mean_intensity_img_hls_arr = np.copy(img_hls_arr)
            subtracted_mean_intensity_img_hls_arr[:,:,1] -= img_sco_hls_mean_arr[:,:,1]

            # Scale subtracted_hsv_mean_intensity between 0 and 255 (HSV's "V" channel)
            (minus_hsv_mean_intensity_min, minus_hsv_mean_intensity_max) = np.min(subtracted_mean_intensity_img_hsv_arr[:,:,2]), np.max(subtracted_mean_intensity_img_hsv_arr[:,:,2])
            subtracted_mean_intensity_img_hsv_arr[:,:,2] = (subtracted_mean_intensity_img_hsv_arr[:,:,2] - minus_hsv_mean_intensity_min) / (minus_hsv_mean_intensity_max-minus_hsv_mean_intensity_min)*255

            # Scale subtracted_hls_mean_intensity between 0 and 255 (HLS's "L" channel)
            (minus_hls_mean_intensity_min, minus_hls_mean_intensity_max) = np.min(subtracted_mean_intensity_img_hls_arr[:,:,1]), np.max(subtracted_mean_intensity_img_hls_arr[:,:,1])
            subtracted_mean_intensity_img_hls_arr[:,:,1] = (subtracted_mean_intensity_img_hls_arr[:,:,1] - minus_hls_mean_intensity_min) / (minus_hls_mean_intensity_max-minus_hls_mean_intensity_min)*255


            # Convert subtracted_mean_intensity  to RGB
            subtracted_mean_intensity_img_rgb_arr_from_hsv = cv.cvtColor (subtracted_mean_intensity_img_hsv_arr.astype(np.uint8), cv.COLOR_HSV2RGB)
            subtracted_mean_intensity_img_rgb_arr_from_hls = cv.cvtColor (subtracted_mean_intensity_img_hls_arr.astype(np.uint8), cv.COLOR_HLS2RGB)

            # Save subtracted_mean_intensity (HSV)
            subtracted_mean_intensity_img_rgb_from_hsv = Image.fromarray( subtracted_mean_intensity_img_rgb_arr_from_hsv )
            dest_full_filename = os.path.join (dest_folder_minus_hsv_mean_insensity,d_subcategory,filename)
            subtracted_mean_intensity_img_rgb_from_hsv.save( dest_full_filename )

            # Save subtracted_mean_intensity (HLS)
            subtracted_mean_intensity_img_rgb_from_hls = Image.fromarray( subtracted_mean_intensity_img_rgb_arr_from_hls )
            dest_full_filename = os.path.join (dest_folder_minus_hls_mean_insensity,d_subcategory,filename)
            subtracted_mean_intensity_img_rgb_from_hls.save( dest_full_filename )

            # Boring to wait
            if img_cnt%100 == 0:
                print ("Processed {} files".format(img_cnt))
            img_cnt += 1



#print (np.min(subtracted_mean_intensity_img_hsv_arr), np.max(subtracted_mean_intensity_img_hsv_arr))
#to_show = np.copy(img_hsv_arr).astype(np.uint8)
#to_show = cv.cvtColor(to_show, cv.COLOR_HSV2RGB)
#plt.imshow(to_show)
#plt.show()
