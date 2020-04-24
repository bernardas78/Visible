# Try to equalize the images by equalizing intensity
#   Src:
#       Mean images of each SCO ("D:\\Visible_Data\\ColorSpaces\\SxoX_mean.jpg)
#       Sample images of each SCO ("D:\\Visible_Data\\ColorSpaces\\SxoX_original.jpg)
#   Dest:
#       RGB=>HSV=>RGB conversion SxoX_hsv.jpg (make sure conversion works)
#       Removed an HSV channel: HSV_Var[Hue|Saturation|Value]\\SxoX_hsv_[Hue|Saturation|Value]_[0|64|128|192|255].jpg
#       Subtracted mean HSV channel value by SCO: SubtractedMean\\scoX_minus_mean_[Hue|Saturation|Value].jpg

from PIL import Image
import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt

# Dest
colorspace_folder = "D:\\Visible_Data\\ColorSpaces\\"

# Create a few sample images (how they look after subtracting means)
do_sample_images = True

for sco in ["1","4"]:

    # Open image
    img_rgb = Image.open( os.path.join(colorspace_folder,"SCO" + sco + "_original.jpg") )
    img_rgb_arr = np.asarray(img_rgb)

    # Open SCO's mean
    img_mean_rgb = Image.open(os.path.join(colorspace_folder, "HSV_Means","SCO" + sco + "_Mean.jpg"))  # hsv mean
    #img_mean_rgb = Image.open( os.path.join(colorspace_folder,"SCO" + sco + "_Mean.jpg") )     #rgb mean
    img_mean_rgb_arr = np.asarray(img_mean_rgb)

    # Convert to HSV
    img_hsv_arr = cv.cvtColor (img_rgb_arr, cv.COLOR_RGB2HSV).astype(np.float) # converting to float because uint8(1)-uint8(2)=255
    img_mean_hsv_arr = cv.cvtColor (img_mean_rgb_arr, cv.COLOR_RGB2HSV).astype(np.float)

    # Variant Hue, Saturation, Value
    for var_index, var_name in enumerate (["Hue", "Saturation", "Value"] ):

        # Create folder if needed
        var_folder = os.path.join(colorspace_folder,"HSV_Var"+var_name)
        if not os.path.exists( var_folder ):
            os.makedirs( var_folder)

        # make = 0,64,128,192,255
        for newval in [0, 64,128,192, 255]:
            img_hsv_arr_removed_value = np.copy (img_hsv_arr)
            # Remove channel
            img_hsv_arr_removed_value[:,:,var_index] = newval
            # Convert back to RGB
            img_rgb_arr_removed_value = cv.cvtColor (img_hsv_arr_removed_value.astype(np.uint8), cv.COLOR_HSV2RGB)
            # Save
            dest_filename = os.path.join(var_folder,"sco" + sco + "_hsv_" + var_name + str(newval) + ".jpg")
            Image.fromarray(img_rgb_arr_removed_value).save(dest_filename)

        # Subtracted SCO means
        img_hsv_arr_subtracted_mean = np.copy(img_hsv_arr)
        # subtract mean sco's value.
        img_hsv_arr_subtracted_mean[:,:,var_index] = img_hsv_arr_subtracted_mean[:,:,var_index] -  img_mean_hsv_arr[:,:,var_index]
        # Make values between 0 and 255
        img_hsv_arr_subtracted_mean[:, :, var_index] = (img_hsv_arr_subtracted_mean[:,:,var_index] + 255)/2


        #if var_index==2:
        #    print (np.min(img_hsv_arr_subtracted_mean[:,:,var_index]), np.max(img_hsv_arr_subtracted_mean[:,:,var_index]))

        #    to_show = np.copy(img_hsv_arr_subtracted_mean).astype(np.uint8)
        #    #to_show = np.copy(img_hsv_arr).astype(np.uint8)
        #    #to_show = np.copy(img_mean_hsv_arr).astype(np.uint8)
        #    to_show[:, :, 0] = 255
        #    to_show[:, :, 1] = 255
        #    # for i in range(25):
        #    #    to_show[ (i*10):((i+1)*10-1),:,2] = i*10
        #    to_show = cv.cvtColor(to_show, cv.COLOR_HSV2RGB)
        #    plt.imshow(to_show)
        #    plt.show()

        # Convert back to RGB
        img_rgb_arr_subtracted_mean = cv.cvtColor(img_hsv_arr_subtracted_mean.astype(np.uint8), cv.COLOR_HSV2RGB)
        # Save
        dest_filename = os.path.join(colorspace_folder, "SubtractedMean" ,"sco" + sco + "_minus_mean_" + var_name+ ".png")
        Image.fromarray(img_rgb_arr_subtracted_mean).save(dest_filename)

        # Fully scaled
        (min_val, max_val) = np.min(img_hsv_arr_subtracted_mean[:, :, var_index]), np.max(img_hsv_arr_subtracted_mean[:, :, var_index])
        img_hsv_arr_subtracted_mean[:, :, var_index] = (img_hsv_arr_subtracted_mean[:, :, var_index] - min_val) / (max_val-min_val) * 255
        # Convert back to RGB
        img_rgb_arr_subtracted_mean = cv.cvtColor(img_hsv_arr_subtracted_mean.astype(np.uint8), cv.COLOR_HSV2RGB)
        # Save
        dest_filename = os.path.join(colorspace_folder, "SubtractedMean" ,"sco" + sco + "_minus_mean_" + var_name+ "_scaled_0to255.jpg")
        Image.fromarray(img_rgb_arr_subtracted_mean).save(dest_filename)

    # Original converted back to to RGB and Save
    img_rgb_arr_transformed = cv.cvtColor (img_hsv_arr.astype(np.uint8), cv.COLOR_HSV2RGB)
    dest_filename = os.path.join(colorspace_folder,"sco" + sco + "_hsv.jpg")
    Image.fromarray(img_rgb_arr_transformed).save(dest_filename)

