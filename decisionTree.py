#Make sure to install Graphviz 
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

#MOST OF THESE FUNCTIONS ARE STILL BEING WORKED ON AND CAN BE LABEL AS IMPRACTICAL OR INEFFICIENT

#Splits the DataSet as inputData and target data
def trainDataset(dataset,Xmax,labelCol):
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    target = dataset.iloc[:,labelCol]
    inputData = dataset.drop(dataset.columns[labelCol],axis=1)

    return inputData,target

#Splits Data into X_train, X_test, y_train, y_test
def trainModels(dataset,Xmax,labelCol,attackNum,inputData,target):    
    # split into input (X) and output (y) variables
    #print("Separating the data from the labels")
    X = inputData
    y = target

    #TRAINING DECISION TREE
    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")
    print("Creating Models")
   
    #Training the decision tree classifier 
    
    #fineTune(model=model2,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name='treeTune2.png',attackNum=attackNum,Xmax=Xmax)
    #fineTune(model=model4,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name='treeTune4.png',attackNum=attackNum,Xmax=Xmax)
    #fineTune(model=model6,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name='treeTune6.png',attackNum=attackNum,Xmax=Xmax)
    #fineTune(model=model8,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name='treeTune8.png',attackNum=attackNum,Xmax=Xmax)

    return X, y,attackNum,Xmax,X_train,y_train,X_test,y_test

#Creates Creates and Saves a .png Image to Data Folder
#Have not Optimize to Save on Specific Folder
def printModels(model,X,X_train,y_train,X_test,y_test,name):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=X.columns,  
                      class_names=True, node_ids=True,
                      filled=True, rounded=True,
                      special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png(name)
    Image(graph.create_png())

    y_pred =  model.predict(X_test)
    print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=model.predict(X_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))

#Creates Models Based on the Tree Depth Number
def CreateModel(cntr,X,X_train,y_train,X_test,y_test,name):
    model = DecisionTreeClassifier(criterion='entropy', random_state=0,
                                        #max_features=attackNum,max_leaf_nodes=Xmax 
                                        max_depth=cntr)
    print("Classifier Done")
    
    #Training the decision tree classifier 
    model.fit(X_train,y_train)
    print("Training Decision Tree Done")
    printModels(model=model,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name)
    print("Training Complete")

#Predicts the Attack Based on Row Input
def predictM(rowNum, model,):
    pred = model.predict(rowNum)
    print("Predicted Outcome: ",pred)
    print()

#Does Not Really Fine Tune, Just Modifies Hyperperameters for DecisionTreeClassifier
def fineTune(X, model,attackNum,Xmax,X_train,y_train,X_test,y_test,name):
    print("Tuning Decision Tree...")
    model = DecisionTreeClassifier(criterion='entropy', min_samples_split=50, 
                                    random_state=0, max_features=attackNum,
                                    max_leaf_nodes=Xmax )
    model.fit(X_train, y_train)
    print("Tuning Decision Tree Done")
    printModels(model=model,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name)


    #return X, model
