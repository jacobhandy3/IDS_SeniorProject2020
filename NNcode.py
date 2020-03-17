from numpy import loadtxt
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import tensorflow as tf
import keras as k
import glob as glob
import os
import matplotlib.pyplot as plt

'''CONVERT TO FUNCTION BY PASSING FOLDER PATH & DICTIONARY OF ATTACKS'''


'''ANALYSIS OF CICIDS DATASET WITH 8 CSVs '''

def CICAnalysis(path, mapped):
    #open dataset folder path
    os.chdir(path)
    #find files with glob
    fileList = glob.glob("*.csv")
    #create a temp list
    dataList = []
    #loop thorugh the files
    for file in fileList:
        #read each file as csv with pandas
        data = pd.read_csv(file, header=0)
        #append to temp list
        dataList.append(data)
    #concat vertically
    dataset = pd.concat(dataList, axis=0)
    #replace all labels with a numeral assignment
    for key in mapped:
        dataset = dataset.replace(to_replace=key, value=mapped[key])
    #replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    #drop any row with a NaN
    dataset = dataset.dropna(how="any")
    # split into input (X) and output (y) variables
    colNum = dataset.shape[1]   #79?
    X = dataset.iloc[:,0:(colNum-2)]
    y = dataset.iloc[:,(colNum-1)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    print("Now onto the ML code")
    # define the keras model
    model = Sequential()
    model.add(Dense(39, input_dim=(colNum-2), activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='softmax'))

    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    bs = 25
    # fit the keras model on the dataset
    hist = model.fit(X, y, epochs=50, batch_size=bs, validation_data=(X_test,y_test))
    #summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def UNSWAnalysis(path, mapped, rowL):
    #open dataset folder path
    os.chdir(path)
    #find files with glob
    fileList = glob.glob("*.csv")
    #create a temp list
    dataList = []
    #loop thorugh the files
    for file in fileList:
        #read each file as csv with pandas
        data = pd.read_csv(file, header=None, index_col=None, )
        #append to temp list
        dataList.append(data)
    #concat vertically
    dataset = pd.concat(dataList, axis=0)
    dataset.drop([1,3], axis=1, inplace=True)
    dataset[47].fillna(value="Benign", inplace=True)
    mapping = AddToMap(mapped, dataset, rowL)
    #replace all labels with a numeral assignment
    print("Begin replacing...")
    for key in mapping:
        dataset.replace(to_replace=key, value=mapping[key], inplace=True)
    #replace the Infinity values with NaNs
    print("done")
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    #drop any row with a NaN
    dataset = dataset.dropna(how="any")
    # split into input (X) and output (y) variables
    colNum = dataset.shape[1]   #79?
    X = dataset.iloc[:,0:(colNum-3)]
    y = dataset.iloc[:,(colNum-2)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")
    # define the keras model
    model = Sequential()
    model.add(Dense(39, input_dim=(colNum-3), activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    bs = 25
    # fit the keras model on the dataset
    hist = model.fit(X, y, epochs=50, batch_size=bs, validation_data=(X_test,y_test))
    #summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def AddToMap(m, ds, rL):
    i = 0
    for r in rL:
        for cell in ds.iloc[:,r]:
            if cell in m:
                continue
            else:
                m[cell] = i
                i+=1
        i = 0
    return m