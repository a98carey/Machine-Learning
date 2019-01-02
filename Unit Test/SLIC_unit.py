import cv2
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

def SLIC(src ,n_segments, compactness):

    # lab =  cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # src =  cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    segments = slic(src, n_segments, compactness)
    dst = mark_boundaries(src, segments)
    # plt.subplot(121)
    # plt.title("n_segments=20")
    # plt.imshow(segments)
    # segments2 = slic(img, n_segments=300, compactness=10)
    # out2=mark_boundaries(img,segments2)
    # plt.subplot(122)
    # plt.title("n_segments=300")
    # plt.imshow(out2)
    # plt.show()
    return dst

if __name__ == '__main__':
    src = cv2.imread('./eli_walk3.png')
    clone1 = src.copy()
    clone2 = src.copy()
    clone3 = src.copy()
    dst1 = SLIC(clone1, n_segments=300, compactness=10)
    dst2 = SLIC(clone2, n_segments=300, compactness=20)
    dst3 = SLIC(clone3, n_segments=300, compactness=5)

    cv2.imshow('dst1', cv2.pyrUp(dst1))
    cv2.imshow('dst2', cv2.pyrUp(dst2))
    cv2.imshow('dst3', cv2.pyrUp(dst3))
    cv2.waitKey(0)
    cv2.destroyAllWindows()