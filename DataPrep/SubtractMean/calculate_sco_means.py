# Calculate means of dataset and save as images

#   Src: "D:\\Visible_Data\\2.Cropped"
#        "D:\\Visible_Data\\2.Cropped_BySCO"
#   Dest: "D:\\Visible_Data\\Means_Data\\"

do_overall_subcat_mean = True
do_sco_mean = True
do_sample_images = True

from PIL import Image
import numpy as np
import os

# Src
cropped_folder = "D:\\Visible_Data\\2.Cropped"
cropped_folder_by_sco = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
means_folder = "D:\\Visible_Data\\Means_Data\\"


if do_overall_subcat_mean:
    # Init overall mean
    img_mean_arr = np.zeros( (360,360,3), dtype=np.float )
    img_cnt = 0

    for root, d_subcats, f in os.walk(cropped_folder):
        for d_subcat in d_subcats:

            # Init per-category mean
            img_subcat_mean_arr = np.zeros( (360,360,3), dtype=np.float )
            img_subcat_cnt = 0

            for _, _, img_filenames in os.walk( os.path.join( root, d_subcat ) ):
                for img_filename in img_filenames:
                    full_img_filename = os.path.join( root, d_subcat, img_filename )
                    #print (full_img_filename)
                    img = Image.open(full_img_filename)
                    img_arr = np.asarray(img)

                    # update overall mean image
                    img_mean_arr = (img_mean_arr*img_cnt + img_arr ) / (img_cnt+1)
                    img_cnt +=1

                    # update subcat mean image
                    img_subcat_mean_arr = (img_subcat_mean_arr*img_subcat_cnt + img_arr ) / (img_subcat_cnt+1)
                    img_subcat_cnt +=1

                    # It's boring to wait
                    if img_cnt%100==0:
                        print ("Processed {} images".format(img_cnt))
            # Save category mean
            img_subcat_mean = Image.fromarray(img_subcat_mean_arr.astype(np.uint8))
            img_subcat_mean.save( os.path.join ( means_folder, "Subcats", d_subcat+"_Mean.jpg") )

    # Save overall mean
    img_mean = Image.fromarray(img_mean_arr.astype(np.uint8))
    img_mean.save( os.path.join ( means_folder, "SCO", "Overall_Mean.jpg") )

if do_sco_mean:

    for d_sco in os.listdir(cropped_folder_by_sco):
        # Init per-sco mean
        img_sco_mean_arr = np.zeros( (360,360,3), dtype=np.float )
        img_sco_cnt = 0

        for d_subcat in os.listdir( os.path.join(cropped_folder_by_sco, d_sco) ):

            for img_filename in os.listdir( os.path.join( cropped_folder_by_sco, d_sco, d_subcat ) ):

                full_img_filename = os.path.join( cropped_folder_by_sco, d_sco, d_subcat, img_filename )
                #print (full_img_filename)
                img = Image.open(full_img_filename)
                img_arr = np.asarray(img)

                # update sco mean image
                img_sco_mean_arr = (img_sco_mean_arr*img_sco_cnt + img_arr ) / (img_sco_cnt+1)
                img_sco_cnt +=1

                # It's boring to wait
                if img_sco_cnt%100==0:
                    print ("Processed {} images of {}".format(img_sco_cnt, d_sco))

        # Save SCO mean
        img_sco_mean = Image.fromarray(img_sco_mean_arr.astype(np.uint8))
        img_sco_mean.save( os.path.join ( means_folder, "SCO", d_sco+"_Mean.jpg") )

# Subtract mean - to see sample images
if do_sample_images:
    for sco in ["1","4"]:

        img_sco = Image.open( os.path.join(means_folder,"SCO" + sco + "_original.jpg") )
        img_sco_arr = np.asarray(img_sco)

        img_sco_mean = Image.open( os.path.join(means_folder,"SCO", "SCO" + sco + "_Mean.jpg") )
        img_sco_mean_arr = np.asarray(img_sco_mean)

        # Subtracted sco mean image
        img_sco_minus_mean = img_sco_arr.astype(np.float) - img_sco_mean_arr.astype(np.float)
        # Don't scale, just make sure between 0 and 255
        img_sco_minus_mean_unscaled = ((img_sco_minus_mean + 255) / 2).astype(np.uint8)
        Image.fromarray(img_sco_minus_mean_unscaled).save( os.path.join(means_folder, "SampleImages", "sco" + sco + "_minus_sco_mean_unscaled.jpg") )
        # Scale between 0 and 255
        img_sco_minus_mean_scaled = img_sco_minus_mean - np.min(img_sco_minus_mean)
        img_sco_minus_mean_scaled = (img_sco_minus_mean_scaled / np.max(img_sco_minus_mean_scaled) * 255).astype(np.uint8)
        Image.fromarray(img_sco_minus_mean_scaled).save( os.path.join(means_folder, "SampleImages", "sco" + sco + "_minus_sco_mean_scaled_0_to_255.jpg") )

        # Subtracted overall mean image
        img_overall_mean = Image.open( os.path.join(means_folder,"SCO", "Overall_Mean.jpg") )
        img_overall_mean_arr = np.asarray(img_overall_mean)

        img_sco_minus_overall_mean = img_sco_arr.astype(np.float) - img_overall_mean_arr.astype(np.float)
        # Don't scale, just make sure between 0 and 255
        img_sco_minus_overall_mean_unscaled = ((img_sco_minus_overall_mean + 255) / 2).astype(np.uint8)
        Image.fromarray(img_sco_minus_overall_mean_unscaled).save( os.path.join(means_folder, "SampleImages", "sco" + sco + "_minus_overall_mean_unscaled.jpg") )
        # Scale between 0 and 255
        img_sco_minus_overall_mean_scaled = img_sco_minus_overall_mean - np.min(img_sco_minus_overall_mean)
        img_sco_minus_overall_mean_scaled = (img_sco_minus_overall_mean_scaled / np.max(img_sco_minus_overall_mean_scaled) * 255).astype(np.uint8)
        Image.fromarray(img_sco_minus_overall_mean_scaled).save( os.path.join(means_folder, "SampleImages", "sco" + sco + "_minus_overall_mean_scaled_0_to_255.jpg") )
