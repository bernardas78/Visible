
import shutil
import numpy as np
import cv2 as cv
import os
from PIL import Image
from stat import *

bgrnd_path = os.environ['GDRIVE'] + "\\PhD_Data\\Bgrnd"
# Src
src_dir = "D:\\Visible_Data\\2.Cropped_BySCO"
# Dest
masks_dir = "D:\\Visible_Data\\2.Cropped_BySCO_BgrndMasks"
dest_dir = "D:\\Visible_Data\\2.Cropped_BySCO_BgrndRemoved"

def learn_background ( backSub, bgrnd_path):
    for file_or_dir in os.listdir (bgrnd_path):
        full_file_or_dir_path = os.path.join ( bgrnd_path, file_or_dir )
        if S_ISDIR ( os.stat(full_file_or_dir_path)[ST_MODE] ):
            learn_background ( backSub, full_file_or_dir_path)
        else:
            full_filename = os.path.join ( bgrnd_path, full_file_or_dir_path)
            img = np.asarray(Image.open(full_filename))
            backSub.apply(img)

# Boring to wait
img_cnt = 0

for sco_dir in os.listdir(bgrnd_path):

    # Background pics path
    sco_bgrnd_path = os.path.join( bgrnd_path, sco_dir )

    # Learn background of this SCO (using best params from grid search)
    backSub = cv.createBackgroundSubtractorKNN( dist2Threshold=1600, detectShadows=False)
    backSub.setkNNSamples(1)
    backSub.setNSamples(14)
    learn_background(backSub, sco_bgrnd_path)

    # Source SCO dir
    sco_src_dir = os.path.join (src_dir, sco_dir)

    # Dest masks, masked-pics dirs
    sco_masks_dir = os.path.join( masks_dir, sco_dir)
    sco_dest_dir = os.path.join( dest_dir, sco_dir)

    # Remove entire SCO - masks
    if os.path.exists(sco_masks_dir):
        shutil.rmtree(sco_masks_dir)
    os.makedirs(sco_masks_dir)

    # Remove entire SCO - removed bgrnd
    if os.path.exists(sco_dest_dir):
        shutil.rmtree(sco_dest_dir)
    os.makedirs(sco_dest_dir)


    for sco_subcat_src_dir in os.listdir (sco_src_dir):

        # Create mask dirs
        mask_subcat_path = os.path.join(sco_masks_dir, sco_subcat_src_dir)
        os.makedirs( mask_subcat_path )

        # Create removed-bgrnd dirs
        rmvd_bgrnd_subcat_path = os.path.join(sco_dest_dir, sco_subcat_src_dir)
        os.makedirs( rmvd_bgrnd_subcat_path )

        # Src dir
        src_subcat_path = os.path.join( sco_src_dir, sco_subcat_src_dir)

        for img_filename in os.listdir ( src_subcat_path ):

            img_path = os.path.join( src_subcat_path, img_filename)
            img = np.asarray(Image.open(img_path))

            fgMask = backSub.apply(img, learningRate=1e-8)

            #make image to contain both orig and mask side by side
            img_both_arr = np.zeros ( (fgMask.shape[0],fgMask.shape[1]*2,3), dtype=np.uint8 )
            img_both_arr[:, fgMask.shape[1]:, : ] = np.array(img)
            img_both_arr[:, :fgMask.shape[1], : ] = np.expand_dims( np.array(fgMask), axis=-1)

            # Write mask+orig
            mask_filename = os.path.join( mask_subcat_path, img_filename)
            img_both = Image.fromarray(img_both_arr, mode='RGB' )
            img_both.save ( mask_filename )
            #cv.imwrite(mask_filename,  img_both )

            # Remove background
            img_removedBgrnd_arr = np.bitwise_and ( np.array(img), np.expand_dims( np.array(fgMask), axis=-1) )

            # Write removed background image
            rmvdBgrnd_filename = os.path.join( rmvd_bgrnd_subcat_path, img_filename)
            img_removedBgrnd = Image.fromarray(img_removedBgrnd_arr, mode='RGB')
            img_removedBgrnd.save ( rmvdBgrnd_filename )
            #cv.imwrite(rmvdBgrnd_filename, Image.fromarray( img_removedBgrnd_arr, mode='RGB' ) )

            # Boring to wait
            if img_cnt%100 == 0:
                print ("Processed {} files".format(img_cnt))
            img_cnt += 1

            #break
        #break
    #break