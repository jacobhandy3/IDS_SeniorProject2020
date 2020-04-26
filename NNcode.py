from numpy import loadtxt
from keras.optimizers import Adamax
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
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

def NNanalysis(path, dataset, Xmax, labelCol, attackNum):
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = dataset.iloc[:,0:Xmax]
    y = dataset.iloc[:,labelCol]
    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")
    # define the keras model
    model = Sequential()
    model.add(Dense(30, input_dim=Xmax, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(attackNum, activation='relu'))
    model.add(Dense(attackNum, activation='softmax'))

    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    bs = 25
    # fit the keras model on the dataset
    hist = model.fit(X, y, epochs=25, batch_size=bs, validation_data=(X_test,y_test))
    ACCf = path + "\ACCxEPOCH.png"
    LOSSf = path + "\LOSSxEPOCH.png"
    #summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.savefig(ACCf)
    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(LOSSf)

#takes existing map, pandas data frame, and list of columns to review
def AddToMap(m, ds, cL):
    i = 0
    #for each column in list
    for c in cL:
        #for each cell in the column
        for cell in ds.iloc[:,c]:
            #if already in map, do nothing
            if cell in m:
                continue
            #if not in map, add to map with i
            else:
                m[cell] = i
                i+=1
        #reset counter for next column
        i = 0
    #return the map
    return m

#takes a folde path to find csv files, 0 for a header and None for no header
#and 0 for an index column or None for no index column
def formatData(path, head, indexCol):
    #open dataset folder path
    os.chdir(path)
    #find files with glob
    fileList = glob.glob("*.csv")
    #create a temp list
    dataList = []
    #loop thorugh the files
    for file in fileList:
        #read each file as csv with pandas
        data = pd.read_csv(file, header=head, index_col=indexCol)
        #append to temp list
        dataList.append(data)
    #concat vertically
    dataset = pd.concat(dataList, axis=0)
    #return the dataset
    return dataset

#takes a dataset, list of value to replace missing data in
#corresponding column in missCols
def missData(dataset, miss, missCols):
    i = 0
    #for each column in list of columns, missCols
    for col in missCols:
        #With the column, fill any missing data with item in miss list
        dataset[col].fillna(value=miss[i], inplace=True)
        #increment counter for next item to use for next column
        i+=1
    #return the dataset
    return dataset