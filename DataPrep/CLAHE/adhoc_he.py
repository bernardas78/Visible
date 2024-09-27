# Histogram equalization of a file
#   Tutorial: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
#   Open CV:

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

show_he_pics = True
show_he_histograms = True
show_ahe = True
show_clahe = True

#filename = "D:\\Visible_Data\\2.Cropped\\4\\18002_8_20191127200639585.jpg" #agurkas
#filename = "D:\\Visible_Data\\2.Cropped_BySCO\\SCO2\\4\\00000058007_7_20190909174648209.jpg" #neapsviestos vynuoges

#histEqual_filename = "D:\\Visible_Data\\temp\\histEqual_18002_8_20191127200639585.jpg"

path = r"G:/My Drive/VU/Dise/img/he"

#filename = os.path.join (path, "he_grapes_orig.png") #neapsviestos vynuoges
#hist_filename = os.path.join (path, "he_grapes_orig_hist.png")
#histEqual_filename = os.path.join (path, "he_grapes_he.png")
#ahe_filename = os.path.join (path, "he_grapes_ahe.png")
#clahe_filename = os.path.join (path, "he_grapes_clahe.png")

filename = os.path.join (path, "he_cucum_orig.png") #agurkas
hist_filename = os.path.join (path, "he_cucum_orig_hist.png")
histEqual_filename = os.path.join (path, "he_cucum_he.png")
ahe_filename = os.path.join (path, "he_cucum_ahe.png")
clahe_filename = os.path.join (path, "he_cucum_clahe.png")


#filename = "D:\\Visible_Data\\2.Cropped_BySco\\SCO2\\4\\00001154371_6_20190902183714763.jpg"
#histEqual_filename = "D:\\Visible_Data\\temp\\histEqual_00001154371_6_20190902183714763.jpg"

src = cv.imread(filename)

# Equalize grey histogram
src_grey = cv.cvtColor( src, cv.COLOR_BGR2GRAY)
dest_grey = cv.equalizeHist( src_grey)
#cv.imshow('Source image', src)
#cv.imshow('Equalized Image', dest_grey)

src_hsv = cv.cvtColor( src, cv.COLOR_BGR2HSV)
dest_hsv = np.copy(src_hsv)

# Equalize HSV's "V" channel

if show_he_pics:
    dest_hsv[:,:,2] = cv.equalizeHist( src_hsv[:,:,2])
    # convert back to BGR
    dest = cv.cvtColor( dest_hsv, cv.COLOR_HSV2BGR)
    #cv.imshow('Equalized HSV V Image', dest)
    cv.imwrite( histEqual_filename, dest)

# Show histogram before and after AHE

if show_he_histograms:
    hst_vals = src_hsv[:,:,2].ravel() / 255.
    plt.hist(hst_vals, bins=int(len(np.unique(hst_vals))/2), label="Before equalizeHist", rwidth=1, cumulative=True)
    plt.yticks([len(hst_vals), len(hst_vals)/2], ["100%", "50%"], fontsize=20)
    plt.xticks( [0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    #plt.title ("Histogram of intensity", fontsize=20)
    plt.tight_layout()
    plt.savefig(hist_filename)

    #cv.imwrite(hist_filename, dest)
    #hst_vals = dest_hsv[:,:,2].ravel()
    #plt.hist(hst_vals, bins=len(np.unique(hst_vals)), label="After equalizeHist", rwidth=0.5, cumulative=True)
    #plt.show()


if show_ahe:

    tileGridSize_OneSide = [1, 2, 4, 8, 16, 32]
    (rows, cols) = (2,3)
    (height,width,channels) = src_hsv.shape

    canvas = np.zeros( (height*rows, width*cols, channels), dtype=np.uint8)


    for i,tileGridSize_OneSide in enumerate(tileGridSize_OneSide):
        clahe = cv.createCLAHE(clipLimit=10000, tileGridSize=(tileGridSize_OneSide,tileGridSize_OneSide))
        dest_clahe = np.copy(src_hsv)

        dest_clahe[:,:,2] = clahe.apply(dest_clahe[:,:,2] ) #replacing "V" channel of HSV
        dest_clahe_bgr = cv.cvtColor( dest_clahe, cv.COLOR_HSV2BGR)

        row, col = int(i/cols), i%cols
        print ("curret row {},col {}; width {}, height {}".format(row,col,width,height))
        canvas [ row*height:(row+1)*height, (col*width):(col+1)*width, :] = np.copy(dest_clahe_bgr)
        #cv.putText( canvas, "tileGrid=({},{})".format(tileGridSize_OneSide,tileGridSize_OneSide),
        #            (col*width+80, row*height+350),
        #            fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1, color=[0,0,255])

        del clahe
    #cv.imshow("AHE", canvas)
    cv.imwrite(ahe_filename, canvas)


if show_clahe:

    scale = 0.5
    tileGridSize_OneSide = [2, 4, 8, 16]
    clipLimits = [20, 10, 5, 3] #[1000, 100, 10, 5]
    (rows, cols) = (len(tileGridSize_OneSide), len(clipLimits))

    (height,width,channels) = int(src_hsv.shape[0]*scale) ,int(src_hsv.shape[1]*scale), src_hsv.shape[2]

    canvas = np.zeros( (height*rows, width*cols, channels), dtype=np.uint8)
    for row,tileGridSize_OneSide in enumerate(tileGridSize_OneSide):
        for col, clipLimit in enumerate(clipLimits):
            clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize_OneSide,tileGridSize_OneSide))
            dest_clahe = cv.resize(src_hsv, (height,width))

            dest_clahe[:,:,2] = clahe.apply(dest_clahe[:,:,2] ) #replacing "V" channel of HSV
            dest_clahe_bgr = cv.cvtColor( dest_clahe, cv.COLOR_HSV2BGR)

            #print ("curret row {},col {}".format(row,col))
            canvas [ row*height:(row+1)*height, (col*width):(col+1)*width, :] = np.copy(dest_clahe_bgr)
            cv.putText( canvas, "clipLim={} tileGrd=({},{})".format(clipLimit,tileGridSize_OneSide,tileGridSize_OneSide),
                        (col*width+50, row*height+170),
                        fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=[0,0,255])

            del clahe
    #cv.imshow("CLAHE", canvas)
    cv.imwrite(clahe_filename, canvas)

cv.waitKey()