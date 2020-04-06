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

def NNanalysis(dataset,Xmax,labelCol,attackNum):
  
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = dataset.iloc[:,0:Xmax]
    y = dataset.iloc[:,labelCol]
    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")


    # define the keras model
    model = Sequential()
    model.add(Dense(39, input_dim=Xmax, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(26, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(attackNum, activation='softmax'))

    # compile the keras model
    model.compile(optimizer = Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07),loss='sparse_categorical_crossentropy', metrics =['accuracy'])
    bs = 25
    # fit the keras model on the dataset
    hist = model.fit(X_train, y_train, epochs=50, batch_size=bs, validation_data=(X_test,y_test))
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
    return X, model


def predictM(rowNum, model):
    pred = model.predict(rowNum)
    print(pred)