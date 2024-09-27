from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np

#Image.open("Illumination_diff_zones.png")

src = cv.imread("Illumination_diff_zones.png")

#src_grey = cv.cvtColor( src, cv.COLOR_BGR2GRAY)

src_hsv = cv.cvtColor( src, cv.COLOR_BGR2HSV)

# Value - intensity channel
regions5 = np.uint8( np.uint8(np.copy(src_hsv[:,:,2]) / 256. * 5.) /4 * 255 )

plt.imshow( regions5, cmap='gray')

#plt.hist(src_hsv[:,:,2])
plt.show()

#dest_hsv = np.copy(src_hsv)
