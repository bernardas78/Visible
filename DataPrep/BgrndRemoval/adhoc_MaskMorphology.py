# Mask improving experiments:
#   Remove "stars"
#   Remove white around scanner window
#   Close gaps inside product


import numpy as np
import cv2 as cv
import os
from PIL import Image
from stat import *
#from matplotlib import pyplot as plt

sco1_bgrnd_path = "D:\\Google Drive\\PhD_Data\\Bgrnd\\SCO1"

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


backSub = cv.createBackgroundSubtractorKNN( dist2Threshold=1600, detectShadows=False)
backSub.setkNNSamples(1)
backSub.setNSamples(14)
learn_background(backSub, sco1_bgrnd_path)

#sample_img_path = 'D:\\Visible_Data\\BgrndRemoval\\00001158356_4_20190916213946561.jpg'
sample_img_path = 'D:\\Visible_Data\\BgrndRemoval\\00001158745_10_2019092013030834.jpg'
sample_img = np.asarray ( Image.open(sample_img_path) )


fgMask = backSub.apply(sample_img, learningRate=1e-8) # default value for no-learning learningRate=0 fails; need a small value
#cv.imshow("Mask", fgMask)

(width, height) = fgMask.shape

canvas = np.zeros ( (height, width*4), np.uint8)

# Mask
col,row=0,0
canvas [row*height:(row+1)*height, col*width:(col+1)*width ] = fgMask
cv.putText( canvas, "Original mask",
                    (col*width+50, row*height+50),
                    fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=255)

# MORPHOLOGY
#   Remove stars
#   Remove while around scanner window
#   Close gaps within product


# Opened masks (various kernel size)
for i,kernelSize in enumerate([1,2,3]):
    kernel = np.ones((kernelSize,kernelSize),np.uint8)
    fgMaskOpened = cv.morphologyEx( src=fgMask, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
    col,row=i+1,0
    canvas [row*height:(row+1)*height, col*width:(col+1)*width ] = fgMaskOpened
    cv.putText( canvas, "Opened mask (kernel={})".format(kernel.shape[0]),
                        (col*width+50, row*height+50),
                        fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=255)

cv.imshow("Opened Mask", canvas)

# Opened THEN Closed masks (opened using kernelsize=2; closing using various kernel size)
openingkernel = np.ones((2,2),np.uint8)
fgMaskOpened = cv.morphologyEx( src=fgMask, op=cv.MORPH_OPEN, kernel=openingkernel, iterations=1)
col,row=0,0
canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = fgMaskOpened
cv.putText(canvas, "Opened mask (kernel={})...".format(openingkernel.shape[0]),
           (col * width + 50, row * height + 50),
           fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=255)

for i,closingkernelSize in enumerate([3,6,9]):
    closingkernel = np.ones((closingkernelSize,closingkernelSize),np.uint8)
    fgMaskOpenedClosed = cv.morphologyEx( src=fgMaskOpened, op=cv.MORPH_CLOSE, kernel=closingkernel, iterations=1)
    col,row=i+1,0
    canvas [row*height:(row+1)*height, col*width:(col+1)*width ] = fgMaskOpenedClosed
    cv.putText( canvas, "...then closed mask (kernel={})".format(closingkernel.shape[0]),
                        (col*width+50, row*height+50),
                        fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=255)

cv.imshow("Closed Mask", canvas)

cv.waitKey(0)