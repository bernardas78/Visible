# Prepares dataset by subtracting SCO's mean picture
#   uses: pre-calculated means for each SCO (

#   Src: "D:\\Visible_Data\\2.Cropped_BySCO"
#   Dest: "D:\\Visible_Data\\2a.Subtracted_SCO_Mean"

from PIL import Image
import os
import numpy as np

# Src
cropped_folder = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
dest_folder = "D:\\Visible_Data\\2a.Subtracted_SCO_Mean"
# Means folder
means_folder = "D:\\Visible_Data\\Means_Data\\"

# counter
img_cnt = 0

for d_sco in os.listdir(cropped_folder):
    #print ("Here is a new SCO: {}".format(d_sco))

    # Load SCO's mean image
    img_sco_mean = Image.open(os.path.join(means_folder, "SCO", d_sco + "_Mean.jpg"))
    img_sco_mean_arr = np.asarray(img_sco_mean).astype(np.float)

    for d_subcategory in os.listdir( os.path.join(cropped_folder, d_sco) ):

        # Create subcat dir if needed
        subcat_dir_fullpath = os.path.join (dest_folder,d_subcategory)
        if not os.path.exists(subcat_dir_fullpath):
            os.makedirs(subcat_dir_fullpath)

        for filename in os.listdir( os.path.join(cropped_folder, d_sco, d_subcategory) ):

            # Open each image
            img = Image.open ( os.path.join (cropped_folder,d_sco, d_subcategory, filename) )
            img_arr = np.asarray (img).astype(np.float)

            # Subtract mean
            img_minus_sco_mean_arr = img_arr - img_sco_mean_arr

            # Scale between 0 and 255
            img_minus_sco_mean_arr_scaled = img_minus_sco_mean_arr - np.min(img_minus_sco_mean_arr)
            img_minus_sco_mean_arr_scaled = (img_minus_sco_mean_arr_scaled / np.max(img_minus_sco_mean_arr_scaled) * 255).astype(np.uint8)

            # Save de-meaned file
            img_minus_sco_mean_scaled = Image.fromarray( img_minus_sco_mean_arr_scaled )
            dest_full_filename = os.path.join (dest_folder,d_subcategory,filename)
            img_minus_sco_mean_scaled.save( dest_full_filename )

            # Boring to wait
            if img_cnt%100 == 0:
                print ("Processed {} files".format(img_cnt))
            img_cnt += 1
