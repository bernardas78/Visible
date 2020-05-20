
import numpy as np
import cv2 as cv
import os
from PIL import Image
from stat import *
#from matplotlib import pyplot as plt

sco1_bgrnd_path = "D:\\Google Drive\\PhD_Data\\Bgrnd\\SCO1"
#sco1_bgrnd_path = "D:\\Visible_Data\\2.Cropped_BySCO\\SCO1"

do_param_values = False
do_grid_search=False
do_all_sco1_pics=True


def learn_background ( backSub, sco1_bgrnd_path):
    for file_or_dir in os.listdir (sco1_bgrnd_path):
        full_file_or_dir_path = os.path.join ( sco1_bgrnd_path, file_or_dir )
        if S_ISDIR ( os.stat(full_file_or_dir_path)[ST_MODE] ):
            learn_background ( backSub, full_file_or_dir_path)
        else:
            full_filename = os.path.join ( sco1_bgrnd_path, full_file_or_dir_path)
            img = np.asarray(Image.open(full_filename))
            backSub.apply(img)




backSub = cv.createBackgroundSubtractorKNN( detectShadows =False)

learn_background ( backSub, sco1_bgrnd_path )

#sample_img_path = 'D:\\Visible_Data\\BgrndRemoval\\00001158356_4_20190916213946561.jpg'
sample_img_path = 'D:\\Visible_Data\\BgrndRemoval\\00001158745_10_2019092013030834.jpg'

sample_img = np.asarray ( Image.open(sample_img_path) )

if do_param_values:
    dist2Thresh_path = "D:\\Visible_Data\\BgrndRemoval\\Dist2Thresh"
    dist2Threshold_values = np.square ( np.arange(100) )        #default=400
    for dist2Threshold in dist2Threshold_values:
        backSub.setDist2Threshold ( dist2Threshold )
        fgMask = backSub.apply(sample_img, learningRate=1e-8)  # learningRate=0  ==>  learning not applied; 0 doesn't work!!!
        mask_filename = os.path.join ( dist2Thresh_path, str(dist2Threshold) + ("_default" if dist2Threshold==400 else "") + ".jpg")
        cv.imwrite( mask_filename, fgMask )
        #print (mask_filename)

    GMM_M_path = "D:\\Visible_Data\\BgrndRemoval\\GMM_M_Count"
    m_count_values = np.arange(15)*3+1     #default = 7
    for m_count in m_count_values:
        backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
        backSub.setNSamples(m_count)
        learn_background(backSub, sco1_bgrnd_path)

        fgMask = backSub.apply(sample_img, learningRate=1e-8)  # learningRate=0  ==>  learning not applied; 0 doesn't work!!!
        mask_filename = os.path.join ( GMM_M_path, str(m_count) + ("_default" if m_count==7 else "") + ".jpg")
        cv.imwrite( mask_filename, fgMask )
        #print (mask_filename)

    KNN_K_path = "D:\\Visible_Data\\BgrndRemoval\\KNN_K_Count"
    k_count_values = np.arange(7)     #default = 2
    for k_count in k_count_values:
        backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
        backSub.setkNNSamples(k_count)
        learn_background(backSub, sco1_bgrnd_path)

        fgMask = backSub.apply(sample_img, learningRate=1e-8)  # learningRate=0  ==>  learning not applied; 0 doesn't work!!!
        mask_filename = os.path.join ( KNN_K_path, str(k_count) + ("_default" if k_count==2 else "") + ".jpg")
        cv.imwrite( mask_filename, fgMask )
        #print (mask_filename)


if do_grid_search:
    #grid search
    grid_search_path = "D:\\Visible_Data\\BgrndRemoval\\gridSearch"
    k_count_values = [0,1,2]
    m_count_values = [7,14,28]
    dist2Threshold_values = [400, 1600, 6400]
    for k_count in k_count_values:
        for m_count in m_count_values:
            for dist2Threshold in dist2Threshold_values:
                backSub = cv.createBackgroundSubtractorKNN( dist2Threshold=dist2Threshold, detectShadows=False)
                backSub.setkNNSamples(k_count)
                backSub.setNSamples(m_count)
                learn_background(backSub, sco1_bgrnd_path)

                fgMask = backSub.apply(sample_img, learningRate=1e-8)
                mask_filename = os.path.join(grid_search_path, "K"+str(k_count) + "_M"+str(m_count) + "_Dist"+str(dist2Threshold) +  ".jpg")
                cv.imwrite(mask_filename, fgMask)


if do_all_sco1_pics:
    # process all SCO1
    backSub = cv.createBackgroundSubtractorKNN( dist2Threshold=1600, detectShadows=False)
    backSub.setkNNSamples(1)
    backSub.setNSamples(14)
    learn_background(backSub, sco1_bgrnd_path)

    sco1_pics = "D:\\Visible_Data\\2.Cropped_BySCO\\SCO1"
    masks_folder = "D:\\Visible_Data\\2.Cropped_BySCO_Masks\\SCO1"

    for subcat_dir in os.listdir (sco1_pics):

        mask_subcat_path = os.path.join(masks_folder, subcat_dir)
        if not os.path.exists( mask_subcat_path ):
            os.makedirs( mask_subcat_path )

        subcat_dir_fullpath = os.path.join( sco1_pics, subcat_dir)
        for img_filename in os.listdir ( subcat_dir_fullpath ):

            img_path = os.path.join( subcat_dir_fullpath, img_filename)
            img = np.asarray(Image.open(img_path))

            fgMask = backSub.apply(img, learningRate=1e-8)

            #make image to contain both orig and mask side by side
            img_both_arr = np.zeros ( (fgMask.shape[0],fgMask.shape[1]*2,3) )
            img_both_arr[:, fgMask.shape[1]:, : ] = np.array(img)
            img_both_arr[:, :fgMask.shape[1], : ] = np.expand_dims( np.array(fgMask), axis=-1)

            mask_filename = os.path.join( mask_subcat_path, img_filename)
            cv.imwrite(mask_filename, img_both_arr)
