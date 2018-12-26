import numpy as np

#----------------------- Function -----------------------#
def suppress(boxes, overlapThresh=0.5):
    '''
    suppress(boxes[, overlapThresh]) -> pick
    非極大值抑制.

    @param boxes         : bounding boxes
    @param overlapThresh : overlap threshold

    @return be picked boxes.
    '''
    # if there are no boxes, return an empty list
    # rects = np.asarray(rects)
    if len(boxes) == 0:
        # print('boxes not found.')
        return boxes

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    tlX   = boxes[:, 0]
    tlY   = boxes[:, 1]
    w     = boxes[:, 2]
    h     = boxes[:, 3]
    score = boxes[:, 4].astype('uint8')
    brX   = tlX + w
    brY   = tlY + h

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area  = w * h
    idxs = np.argsort(score)#[::-1]
    # print('- score :', score)
    # print('- idxs  :', idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # print('- pick  :', pick)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        tlx = np.maximum(tlX[i], tlX[idxs[:last]])
        tly = np.maximum(tlY[i], tlY[idxs[:last]])
        brx = np.minimum(brX[i], brX[idxs[:last]])
        bry = np.minimum(brY[i], brY[idxs[:last]])

        # compute the width and height of the bounding box
        olpW = np.maximum(0, brx - tlx + 1)
        olpH = np.maximum(0, bry - tly + 1)

        # compute the ratio of overlap
        overlap = (olpW * olpH) / area[idxs[:last]]
        # print('- overlap :', overlap)
        # overlap1 = (olpW * olpH) / area[idxs[:last]]
        # overlap2 = (olpW * olpH) / area[idxs[last]]
        # overlap = np.maximum(overlap1, overlap2)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))
        # print('- idxs :', idxs)
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]


if __name__ == '__main__':
    import cv2

    boxes = [[ 14,  79,  71, 261,   9],
             [ 11,  42,  74, 304,   3],
             [  8,   0,  14,  73,   1],
             [ 16,  98,  65, 237,  11],
             [137,   0,  30,  64,   2],
             [114, 309,  57, 113,   4],
             [111, 308,  73, 105,   3],
             [ 15,  72,  71, 269,   7],
             [ 13,  49,  72, 294,   4],
             [116, 309,  41, 123,   5],
             [ 16,  59,  70, 284,   5],
             [ 15,  65,  71, 277,   6],
             [137,   0,  32,  54,   1],
             [ 14,  75,  71, 266,   8],      
             [  0,  34, 187, 360,   1],
             [ 13,  83,  71, 256,  10],
             [  9,  38, 178, 365,   2]]

    boxes = np.asarray(boxes, dtype=np.int32)

    canvas = np.zeros((375, 188, 3), dtype='uint8')
    
    for box in boxes:
        x, y, w, h = box[:4]
        cv2.rectangle(canvas, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)


    canvasNMS = np.zeros((375, 188, 3), dtype='uint8')
    boxes = suppress(boxes)

    for box in boxes:
        x, y, w, h = box[:4]
        cv2.rectangle(canvasNMS, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)

    cv2.imshow('canvasNMS', canvasNMS)    
    cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()