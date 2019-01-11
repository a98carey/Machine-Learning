import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras import initializers
import os
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from skimage.segmentation import slic,mark_boundaries
from keras.models import load_model
from skimage.feature import local_binary_pattern
def SLIC(src ,n_segments, compactness):
    lab =  cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    segments = slic(lab, n_segments, compactness)
    dst = mark_boundaries(src, segments)
    xMin = []
    yMin = []
    xMax = []
    yMax = []
    for i in range(n_segments):
        idx = np.where(segments == i)
        if len(idx[0]) > 0:
            xMax.append(np.max(idx[0]))
            yMax.append(np.max(idx[1]))
            xMin.append(np.min(idx[0]))
            yMin.append(np.min(idx[1]))
    xMax = np.asarray(xMax)
    yMax = np.asarray(yMax)
    xMin = np.asarray(xMin)
    yMin = np.asarray(yMin)
    return dst, xMin, yMin, xMax, yMax

def LBP(src):
    src = cv2.resize(src, (60, 80))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    desc = LocalBinaryPatterns(24, 8)
    hist = desc.describe(gray)
    hist = hist.reshape(1, -1)
    return hist

if __name__ == '__main__':   
    oTest = cv2.imread('./book/00030.jpg')
    oTest = cv2.resize(oTest,(600, 800))

    dst, xMin, yMin, xMax, yMax = SLIC(oTest ,n_segments=100 , compactness=50)

    model_class = load_model('model_class.h5')
    model_x = load_model('model_x.h5')
    model_y = load_model('model_y.h5')
    model_dx = load_model('model_dx.h5')
    model_dy = load_model('model_dy.h5')
    classes = []
    x = []
    y = []
    dx = []
    dy = []
    for i in range(len(xMin)):
        hist = LBP(oTest[xMin[i]:xMax[i]+1, yMin[i]:yMax[i]+1])
        prob = model_class.predict(hist)
        # print(prob)
        # flag = 0
        # for p in range(len(prob[0])):
        #     if prob[0][p] > 0.1 and flag == 0:
        classes.append(model_class.predict_classes(hist))
                # flag = 1
        x.append(model_x .predict_classes(hist))
        y.append(model_y .predict_classes(hist))
        dx.append(model_dx.predict_classes(hist))
        dy.append(model_dy.predict_classes(hist))

        # print(i, classes, x, y, dx, dy, xMin[i], yMin[i], xMax[i], yMax[i])

    classes = np.asarray(classes)
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.asarray(dx)
    dy = np.asarray(dy)

    for i in range(4):
        idx = np.where(classes == [i])

        if len(idx[0]) > 1:

            idxMinx = np.argmin(xMin[idx[0]]) # (xMin[a])
            idxMiny = np.argmin(yMin[idx[0]]) # (yMin[a])
            idxMaxx = np.argmax(xMax[idx[0]]) # (xMax[a])
            idxMaxy = np.argmax(yMax[idx[0]]) # (yMax[a])

            tlx = xMin[idx[0][idxMinx]]
            tly = yMin[idx[0][idxMiny]]
            brx = xMax[idx[0][idxMaxx]]
            bry = yMax[idx[0][idxMaxy]]
           
            if i == 0:
                color = (255,0,0)
                cv2.putText(oTest, 'Book-' , (tly,tlx), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.rectangle(oTest, (tly,tlx), (bry,brx), color, 2)
                # cv2.rectangle(oTest, (idxMiny,idxMinx), (idxMaxy,idxMaxx), color, 2)
            if i == 1:
                color = (0,255,0)
                cv2.putText(oTest, 'Mouse-', (tly,tlx), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.rectangle(oTest, (tly,tlx), (bry,brx), color, 2)
                # cv2.rectangle(oTest, (idxMiny,idxMinx), (idxMaxy,idxMaxx), color, 2)

            if i == 2:
                color = (0,0,255)
                cv2.putText(oTest, 'Eraser-', (tly,tlx), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.rectangle(oTest, (tly,tlx), (bry,brx), color, 2)
                # cv2.rectangle(oTest, (idxMiny,idxMinx), (idxMaxy,idxMaxx), color, 2)

            if i == 3:
                color = (0,255,255)
                cv2.putText(oTest, 'Pen-', (tly,tlx), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
                cv2.rectangle(oTest, (tly,tlx), (bry,brx), color, 2)
                # cv2.rectangle(oTest, (idxMiny,idxMinx), (idxMaxy,idxMaxx), color, 2)

    # cv2.imshow('dst', dst)
    # cv2.imshow('oTest', oTest)
    # # cv2.imwrite('oB.jpg', oTest)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        

        