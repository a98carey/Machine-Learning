import numpy as np
import cv2
np.set_printoptions(threshold = np.inf)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
from keras import initializers



if __name__ == '__main__':
    # read image
    filename = './digits.png'  
    src = cv2.imread(filename, 0)

    batch_size = 200
    epochs = 200
    hidden_units = 400
    trainList = []
    testList  = []
    trainLabelList = []
    testLabelList = []

    # split image
    cells = [np.hsplit(row, 100) for row in np.vsplit(src, 50)]

    trainList = [ i[:50] for i in cells ]
    testList  = [ i[50:] for i in cells ]

    trainSet = np.array(trainList, np.float32)
    testSet  = np.array(testList,  np.float32)

    trainSet = trainSet.reshape(2500, 400) 
    testSet  = testSet.reshape(2500,  400) 


    trainSet_normalize = trainSet / 255
    testSet_normalize = testSet / 255

    # produce label
    for i in range(10):
        for j in range(250):
            trainLabelList.append(i) 
            testLabelList.append(i)
    trainLabel = np.array(trainLabelList, np.float32)
    testLabel = np.array(testLabelList, np.float32)

    train_OneHot = np_utils.to_categorical(trainLabel)
    test_OneHot = np_utils.to_categorical(testLabel)

    model = Sequential()

    #將「輸入層」與「隱藏層1」加入模型
    model.add(Dense(units=hidden_units, 
                    input_dim=400, 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    #將「隱藏層2」加入模型
    model.add(Dense(units=hidden_units, 
                    kernel_initializer='normal',
                    activation='relu'))
    #model.add(Dropout(0.5))

    #將「輸出層」加入模型
    model.add(Dense(units=10, 
                    kernel_initializer='normal', 
                    activation='softmax'))
    print(model.summary())

    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='Adam', metrics=['accuracy'])

    train_history=model.fit(x=trainSet_normalize,
                            y=train_OneHot,#validation_split=0.2, #0.2是把訓練資料20%作為驗證資料
                            epochs=epochs, batch_size=batch_size,verbose=2)   #執行30次訓練週期 每一批次200筆資料 verbose=2:顯示訓練過程

    #評估模型準確率                        
    scores = model.evaluate(testSet_normalize, test_OneHot)
    print()
    print('accuracy=',scores[1])

    # prediction=model.predict_classes(Xtest)
    # prediction 