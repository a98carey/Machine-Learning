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

    dst = SLIC(src, n_segments=300, compactness=10)

    cv2.imshow('dst', cv2.pyrUp(dst))
    cv2.waitKey(0)
    cv2.destroyAllWindows()