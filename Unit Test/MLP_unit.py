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

def load_data():
    path = './train/'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img_path = path + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64,128))
        # img = image.load_img(img_path, target_size=image_size)
        # img_array = image.img_to_array(img)
        images.append(image)

        if 'm' in f:
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels

def load_test_data():
    path = './test/'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img_path = path + f
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64,128))
        # img = image.load_img(img_path, target_size=image_size)
        # img_array = image.img_to_array(img)
        images.append(image)

        if 'm' in f:
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels

if __name__ == '__main__':

    data, labels = load_data()
    test_data, test_labels = load_test_data()

    batch_size = 200
    epochs = 10
    hidden_units = 3780

    trainSetList = []
    trainLabelList = []
    testSetList = []
    testLabelList = []

    HOGDescriptor = cv2.HOGDescriptor()
    descriptorLen = HOGDescriptor.getDescriptorSize()

    for label, imgs in enumerate(data):
        descriptors = HOGDescriptor.compute(imgs)
        trainSetList.append(descriptors.ravel())
        trainLabelList.append(label)
        pass

    for label, imgs in enumerate(test_data):
        descriptors = HOGDescriptor.compute(imgs)
        testSetList.append(descriptors.ravel())
        testLabelList.append(label)
        pass

    trainSet   = np.array(trainSetList, np.float32)
    trainLabel = np.array(trainLabelList, np.int32)
    testSet    = np.array(testSetList, np.float32)
    testLabel  = np.array(testLabelList, np.int32)   

    train_OneHot = np_utils.to_categorical(trainLabel)
    test_OneHot = np_utils.to_categorical(testLabel)

    model = Sequential()

    #將「輸入層」與「隱藏層1」加入模型
    model.add(Dense(units=hidden_units, 
                    input_dim=descriptorLen, 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    #將「隱藏層2」加入模型
    model.add(Dense(units=hidden_units, 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    #將「輸出層」加入模型
    model.add(Dense(units=2, 
                    kernel_initializer='normal', 
                    activation='softmax'))
    print(model.summary())

    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='Adam', metrics=['accuracy'])

    train_history=model.fit(x=trainSet,
                            y=train_OneHot,#validation_split=0.2, #0.2是把訓練資料20%作為驗證資料
                            epochs=epochs, batch_size=batch_size,verbose=2)   #執行30次訓練週期 每一批次200筆資料 verbose=2:顯示訓練過程

    #評估模型準確率                        
    scores = model.evaluate(testSet, test_OneHot)
    print()
    print('accuracy=',scores[1])

    # prediction=model.predict_classes(testSet)
    # prediction 