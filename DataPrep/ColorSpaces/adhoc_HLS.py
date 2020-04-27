# Try to equalize the images by equalizing intensity
#   Src:
#       Mean images of each SCO ("D:\\Visible_Data\\ColorSpaces\\ScoX_mean.jpg)
#       Sample images of each SCO ("D:\\Visible_Data\\ColorSpaces\\ScoX_original.jpg)
#   Dest:
#       RGB=>HLS=>RGB conversion ScoX_hls.jpg (make sure conversion works)
#       Removed an HLS channel: HLS_Var[Hue|Saturation|Value]\\ScoX_hls_[Hue|Saturation|Value]_[0|64|128|192|255].jpg
#       Subtracted mean HLS channel value by SCO: SubtractedMean_HLS\\scoX_minus_mean_[Hue|Saturation|Value].jpg

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
    img_mean_rgb = Image.open(os.path.join(colorspace_folder, "HLS_Means","SCO" + sco + "_Mean.jpg"))  # hls mean
    #img_mean_rgb = Image.open( os.path.join(colorspace_folder,"SCO" + sco + "_Mean.jpg") )     #rgb mean
    img_mean_rgb_arr = np.asarray(img_mean_rgb)

    # Convert to HLS
    img_hls_arr = cv.cvtColor (img_rgb_arr, cv.COLOR_RGB2HLS).astype(np.float) # converting to float because uint8(1)-uint8(2)=255
    img_mean_hls_arr = cv.cvtColor (img_mean_rgb_arr, cv.COLOR_RGB2HLS).astype(np.float)

    # Variant Hue, Saturation, Value
    for var_index, var_name in enumerate (["Hue", "Luminance", "Saturation"] ):

        # Create folder if needed
        var_folder = os.path.join(colorspace_folder,"HLS_Var"+var_name)
        if not os.path.exists( var_folder ):
            os.makedirs( var_folder)

        # make = 0,64,128,192,255
        for newval in [0, 64,128,192, 255]:
            img_hls_arr_removed_value = np.copy (img_hls_arr)
            # Remove channel
            img_hls_arr_removed_value[:,:,var_index] = newval
            # Convert back to RGB
            img_rgb_arr_removed_value = cv.cvtColor (img_hls_arr_removed_value.astype(np.uint8), cv.COLOR_HLS2RGB)
            # Save
            dest_filename = os.path.join(var_folder,"sco" + sco + "_hls_" + var_name + str(newval) + ".jpg")
            Image.fromarray(img_rgb_arr_removed_value).save(dest_filename)

        # Subtracted SCO means
        img_hls_arr_subtracted_mean = np.copy(img_hls_arr)
        # subtract mean sco's value.
        img_hls_arr_subtracted_mean[:,:,var_index] = img_hls_arr_subtracted_mean[:,:,var_index] -  img_mean_hls_arr[:,:,var_index]
        # Make values between 0 and 255
        img_hls_arr_subtracted_mean[:, :, var_index] = (img_hls_arr_subtracted_mean[:,:,var_index] + 255)/2

        # Convert back to RGB
        img_rgb_arr_subtracted_mean = cv.cvtColor(img_hls_arr_subtracted_mean.astype(np.uint8), cv.COLOR_HLS2RGB)
        # Save
        dest_filename = os.path.join(colorspace_folder, "SubtractedMean_HLS" ,"sco" + sco + "_minus_mean_" + var_name+ ".png")
        Image.fromarray(img_rgb_arr_subtracted_mean).save(dest_filename)

        # Fully scaled
        (min_val, max_val) = np.min(img_hls_arr_subtracted_mean[:, :, var_index]), np.max(img_hls_arr_subtracted_mean[:, :, var_index])
        img_hls_arr_subtracted_mean[:, :, var_index] = (img_hls_arr_subtracted_mean[:, :, var_index] - min_val) / (max_val-min_val) * 255
        # Convert back to RGB
        img_rgb_arr_subtracted_mean = cv.cvtColor(img_hls_arr_subtracted_mean.astype(np.uint8), cv.COLOR_HLS2RGB)
        # Save
        dest_filename = os.path.join(colorspace_folder, "SubtractedMean_HLS" ,"sco" + sco + "_minus_mean_" + var_name+ "_scaled_0to255.jpg")
        Image.fromarray(img_rgb_arr_subtracted_mean).save(dest_filename)

    # Original converted back to to RGB and Save
    img_rgb_arr_transformed = cv.cvtColor (img_hls_arr.astype(np.uint8), cv.COLOR_HLS2RGB)
    dest_filename = os.path.join(colorspace_folder,"sco" + sco + "_hls.jpg")
    Image.fromarray(img_rgb_arr_transformed).save(dest_filename)

