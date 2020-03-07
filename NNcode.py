from numpy import loadtxt
from keras.optimizers import adam
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

def NNAnalysis(path, attacks):
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
    for key in attacks:
        dataset = dataset.replace(to_replace=key, value=attacks[key])
    #replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    #drop any row with a NaN
    dataset = dataset.dropna(how="any")
    # split into input (X) and output (y) variables
    colNum = dataset.shape[1]   #79?
    X = dataset.iloc[:,0:(colNum-2)]
    y = dataset.iloc[:,(colNum-1)]
    print("Now onto the ML code")
    # define the keras model
    model = Sequential()
    model.add(Dense(39, input_dim=77, activation='relu'))
    model.add(Dense(26, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='softmax'))

    # compile the keras model
    model.compile(optimizer = adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    bs = 25
    # fit the keras model on the dataset
    hist = model.fit(X, y, epochs=50, batch_size=bs, validation_split=0.35)
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