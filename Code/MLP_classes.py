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

np.set_printoptions(threshold=np.inf)

def load_data():

    print('Loading train data ...')
    images = []
    labels = []
    for i in range(19):
        # Book True / label = 0
        path_BT = './train/B'+str(i+1)+'/true/'
        files_BT = os.listdir(path_BT)
        for f in files_BT:
            img_path = path_BT + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(0)
            pass

        # Book False / label = 4
        path_BF = './train/B'+str(i+1)+'/false/'
        files_BF = os.listdir(path_BF)
        for f in files_BF:
            img_path = path_BF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass

        # Mouse True / label = 1
        path_MT = './train/'+str(i)+'/true/'
        files_MT = os.listdir(path_MT)
        for f in files_MT:
            img_path = path_MT + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(1)
            pass

        # Mouse False / label = 4
        path_MF = './train/'+str(i)+'/false/'
        files_MF = os.listdir(path_MF)
        for f in files_MF:
            img_path = path_MF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass

        # Eraser True / label = 2
        path_ET = './train/E'+str(i+1)+'/true/'
        files_ET = os.listdir(path_ET)
        for f in files_ET:
            img_path = path_ET + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(2)
            pass

        # Eraser False / label = 4
        path_EF = './train/E'+str(i+1)+'/false/'
        files_EF = os.listdir(path_EF)
        for f in files_EF:
            img_path = path_EF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass

        # Pen True / label = 3
        path_PT = './train/P'+str(i+1)+'/true/'
        files_PT = os.listdir(path_PT)
        for f in files_PT:
            img_path = path_PT + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(3)
            pass

        # Pen False / label = 4
        path_PF = './train/P'+str(i+1)+'/false/'
        files_PF = os.listdir(path_PF)
        for f in files_PF:
            img_path = path_PF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass
    data = np.array(images)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels)

    return data, labels

def load_test_data():

    print('Loading test data ...')
    images = []
    labels = []
    # Book True / label = 0
    path_BT = './train/B20/true/'
    files_BT = os.listdir(path_BT)
    for f in files_BT:
        img_path = path_BT + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (60,80))
        images.append(image)
        labels.append(0)
        pass

    # Book False / label = 4
    path_BF = './train/B20/false/'
    files_BF = os.listdir(path_BF)
    for f in files_BF:
        img_path = path_BF + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (60,80))
        images.append(image)
        labels.append(4)
        pass

    # Mouse True / label = 1
    path_MT = './train/19/true/'
    files_MT = os.listdir(path_MT)
    for f in files_MT:
        img_path = path_MT + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (60,80))
        images.append(image)
        labels.append(1)
        pass

    # Mouse False / label = 4
    path_MF = './train/19/false/'
    files_MF = os.listdir(path_MF)
    for f in files_MF:
        img_path = path_MF + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (60,80))
        images.append(image)
        labels.append(4)
        pass

    # Eraser True / label = 2
        path_ET = './train/E20/true/'
        files_ET = os.listdir(path_ET)
        for f in files_ET:
            img_path = path_ET + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(2)
            pass

        # Eraser False / label = 4
        path_EF = './train/E20/false/'
        files_EF = os.listdir(path_EF)
        for f in files_EF:
            img_path = path_EF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass

        # Pen True / label = 3
        path_PT = './train/P20/true/'
        files_PT = os.listdir(path_PT)
        for f in files_PT:
            img_path = path_PT + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(3)
            pass

        # Pen False / label = 4
        path_PF = './train/P20/false/'
        files_PF = os.listdir(path_PF)
        for f in files_PF:
            img_path = path_PF + f
            image = cv2.imread(img_path)
            image = cv2.resize(image, (60,80))
            images.append(image)
            labels.append(4)
            pass
    data = np.array(images)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels)

    return data, labels


if __name__ == '__main__':
    data, labels = load_data() 
    test_data, test_labels = load_test_data()
    # print(test_labels)
    trainSetList = []
    testSetList = []

    # settings for LBP
    radius = 3
    n_points = 8 * radius

    desc = LocalBinaryPatterns(24, 8)
    for imgs in data:
        gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        trainSetList.append(hist.ravel())
        pass

    for imgs in test_data:
        gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        testSetList.append(hist.ravel())
        pass

    trainSet = np.array(trainSetList, np.float32)
    testSet  = np.array(testSetList, np.float32)

    batch_size = 2
    epochs = 50
    hidden_units = 500

    model = Sequential()

    model.add(Dense(units=hidden_units, 
                    input_dim=trainSet.shape[1], 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(units=hidden_units, 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Dense(units=labels.shape[1], 
                    kernel_initializer='normal',
                    activation='softmax'))
    print(model.summary())

    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', # categorical
                  optimizer='Adam', metrics=['accuracy'])

    train_history=model.fit(x=trainSet,
                            y=labels,
                            epochs=epochs, batch_size=batch_size,verbose=2)   #執行xx次訓練週期 每一批次xx筆資料 verbose=2:顯示訓練過程

    #評估模型準確率                        
    scores = model.evaluate(testSet, test_labels)
    print('accuracy=',scores[1])

    # model.save('model_class.h5')
    # predictionC = model.predict_classes(testSet)
    # print(predictionC)

    