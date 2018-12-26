import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import os

kernel = np.ones((3,3),np.uint8)

for i in range(1,11,1):

    img_input_path = r"C:/Users/User/Desktop/class details/machine learnings/final/raw/%d.jpg" %(i)

    img_output_path_blur = r"C:/Users/User/Desktop/class details/machine learnings/final/blur/%d.jpg" %(i)

    img_input_path_whiteBalance = r"C:/Users/User/Desktop/class details/machine learnings/final/whiteBalance/%d.jpg" %(i)

    img_input_path_deblur = r"C:/Users/User/Desktop/class details/machine learnings/final/deblur/%d.jpg" %(i)

    img_input_path_sharp = r"C:/Users/User/Desktop/class details/machine learnings/final/sharp/%d.jpg" %(i)


    image = cv2.imread(img_input_path)

    img = cv2.resize(image, (600, 450))

    blur = cv2.GaussianBlur(img,(5,5),0)

    cv2.imshow("blur {}".format(i), blur)

    # cv2.imwrite(img_output_path_blur,blur)

#----------------------------------------------------------------------------------------------------whiteBalance

    targetPos = 150

    histr = cv2.calcHist([blur],[0],None,[256],[0, 256])

    plt.plot(histr)
    # plt.show("histogram.{}".format(i))
    # plt.figure(i)
    
    moment = 0

    for j in range(0, len(histr)):

        moment = moment + histr[j]*j

    grayHistMode = np.uint8(moment/sum(histr))

    blur_copy = blur.copy()

    frameCopy = np.float32(blur_copy)

    idx = np.where(blur <= grayHistMode)

    frameCopy[idx] = np.float32(blur[idx])*targetPos/grayHistMode

    idx = np.where(blur > grayHistMode)

    frameCopy[idx] = np.float32(blur[idx]-grayHistMode)*(256-targetPos)/(256-grayHistMode) + targetPos

        
    EnhancedImg = np.uint8(frameCopy)  

    EnhancedImg_1 = cv2.calcHist([EnhancedImg],[0],None,[256],[0, 256])

    cv2.imshow('new{}'.format(i), EnhancedImg)
    # cv2.imwrite(img_input_path_whiteBalance, EnhancedImg)

    plt.plot(EnhancedImg_1)
    # plt.show("histogram_change.{}".format(i))
    # plt.figure("histogram.{}".format(i+1))
#----------------------------------------------------------------------------------------------------whiteBalance






cv2.waitKey(0)
cv2.destroyAllWindows()