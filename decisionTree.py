#Import Libraries
import pandas as pd
import numpy as np
import glob as glob
import os
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

#Splits Data into training and testing, Returns splitted data
def trainModels(dataset,labelCol):    
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    y = dataset.iloc[:,labelCol]
    X = dataset.drop(dataset.columns[labelCol],axis=1)

    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)

    return X,y,X_train,y_train,X_test,y_test

#Creates Tree Model and Returns predicted Accuracy
def CreateModel(MaxDepth,Criterion,X,X_train,y_train,X_test,y_test,name,attackNames):
    print("Now onto the ML code")
    print("Creating Models")
    model = DecisionTreeClassifier(criterion=Criterion, splitter="best",
                                        max_depth=MaxDepth)
    print("Classifier Done")
    
    #Training the decision tree classifier 
    model.fit(X_train,y_train)
    print("Training Decision Tree Complete")
    #print specific trees, because png files cannot store too much data without sizing it down
    if MaxDepth<10:
        printModels(model=model,X=X,name=name,attackNames=attackNames)
    print("Training Complete")

    #Predicting test Accuracies
    pred =  model.predict(X_test)
    pred_acc= accuracy_score(y_test, pred)
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=model.predict(X_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=pred))

    return pred_acc

#Creates Creates and Saves a .png Image to selected Folder
def printModels(model,X,name,attackNames):
    print("Printing Models")
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X.columns,  
                      class_names=attackNames, node_ids=True,
                      filled=True, rounded=True,
                      special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(name)
    Image(graph.create_png())