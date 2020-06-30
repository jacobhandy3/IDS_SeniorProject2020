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
import matplotlib.pyplot as plt

originalPath = "/media/southpark86/AMG1/School/Spring 2020/Senior Project/"

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
def CreateModel(Criterion,Impurity,MaxDepth,MaxLeaf,MinSampLeaf,MinSampSplit,Test_Data,X,X_train,y_train,X_test,y_test,name,attackNames):
    print("Now onto the ML code")
    print("Creating Models")
    model = DecisionTreeClassifier(criterion=Criterion, min_impurity_decrease=Impurity,
                                    max_depth= MaxDepth, max_leaf_nodes=MaxLeaf,
                                    min_samples_leaf=MinSampLeaf, min_samples_split=MinSampSplit)
    print("Classifier Done")
    
    #Training the decision tree classifier 
    model.fit(X_train,y_train)
    print("Training Decision Tree Complete")
    #print specific trees, because png files cannot store too much data without sizing it down
    #printModels(model=model,X=X,name=name,attackNames=attackNames)

    #print specific trees, because png files cannot store too much data without sizing it down
    if (Test_Data['MaxDepth']!= None ):
        if MaxDepth<11:
            printModels(model=model,X=X,name=name,attackNames=attackNames)
    print("Training Complete")

    #Predicting test Accuracies
    pred =  model.predict(X_test)
    pred_acc= accuracy_score(y_test, pred)

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

    #Runs the Decision Tree File
def Run_DecisionTree(dataset,attackNames,labelCol,dataName,Test_Data,Criterion):
    #Split and Train Data
    X,y,X_train,y_train,X_test,y_test= trainModels(dataset,labelCol)

    #use to plot gini vs entropy accuracies
    x_label = []
    acc_gini = []
    acc_entropy = []

    #Create and Printing Models, Appends predicted accuracies based on Criterion, Appends MaxDepth
    print("STARTIMG LOOP")
    
    #Checks Whether or not it's Testing for Two Parameters
    if(Test_Data['TwoParameters']):
        print("TWO PARAMETERS")
        LoopThis = Test_Data['MaxDepth'] if (Test_Data['MaxDepth'] != None) else (Test_Data['MaxLeaf'] if (Test_Data['MaxLeaf']!= None) else (Test_Data['MinSampLeaf'] if (Test_Data['MinSampLeaf'] != 1) else Test_Data['MinSampSplit']))
        for ImpureLoop in Test_Data['Impurity']:
            x_label = []
            acc_gini = []
            acc_entropy = []
            for LoopParameter in LoopThis:
                for Crit in Criterion :
                    path= originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Criterion_Gini/Impurity_Decreased/Impurity_"+str(ImpureLoop)+"/" if (Crit == 'gini')  else originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Criterion_Entropy/Impurity_Decreased/Impurity_"+str(ImpureLoop)+"/"
                    name = dataName+'_Dataset_Features_Criterion_Gini_'+Test_Data['Name']+'_'+str(LoopParameter)+'_Impurity_'+str(ImpureLoop)+'.png' if (Crit == 'gini') else dataName+'_Dataset_Features_Criterion_Entropy_'+Test_Data['Name']+'_'+str(LoopParameter)+'_Impurity_'+str(ImpureLoop)+'.png'

                    Impurity = ImpureLoop                    
                    MaxDepth = LoopParameter if Test_Data['MaxDepth'] != None else Test_Data['MaxDepth']
                    MaxLeaf = LoopParameter if Test_Data['MaxLeaf'] != None else Test_Data['MaxLeaf']
                    MinSampLeaf = LoopParameter if Test_Data['MinSampLeaf'] != 1 else Test_Data['MinSampLeaf']
                    MinSampSplit = LoopParameter if Test_Data['MinSampSplit'] != 2 else Test_Data['MinSampSplit']

                    name = path+name
                    #Creates Gini Models
                    pred_acc= CreateModel(Criterion=Crit,Impurity=Impurity,MaxDepth=MaxDepth,MaxLeaf=MaxLeaf,
                                                    MinSampLeaf=MinSampLeaf,MinSampSplit=MinSampSplit,Test_Data=Test_Data, X=X,
                                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                    name=name, attackNames=attackNames)
                    if (Crit == 'gini') :
                        acc_gini.append(pred_acc)
                    else:
                        acc_entropy.append(pred_acc)
                x_label.append(LoopParameter)
                print("Loop: "+str(LoopParameter))

            plottingGraphs(x_label=x_label,acc_gini=acc_gini,acc_entropy=acc_entropy,Test_Data=Test_Data,ImpureLoop=ImpureLoop,dataName=dataName)
    else:
        #Test One Parameter at a Time
        print("ONE PARAMETER") 
        LoopThis = Test_Data['MaxDepth'] if (Test_Data['MaxDepth'] != None) else (Test_Data['MaxLeaf'] if (Test_Data['MaxLeaf']!= None) else (Test_Data['MinSampLeaf'] if (Test_Data['MinSampLeaf'] != 1) else ( Test_Data['Impurity'] if (Test_Data['Impurity'] != 0.0) else Test_Data['MinSampSplit'])))
        for LoopParameter in LoopThis:
            for Crit in Criterion :
                path= originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Criterion_Gini/"+Test_Data['Name']+"/" if (Crit == 'gini')  else originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Criterion_Entropy/"+Test_Data['Name']+"/"
                name = dataName+'_Dataset_Features_Criterion_Gini_'+Test_Data['Name']+'_'+str(LoopParameter)+'.png' if (Crit == 'gini')  else dataName+'_Dataset_Features_Criterion_Entropy_'+Test_Data['Name']+'_'+str(LoopParameter)+'.png'
                
                Impurity = LoopParameter if (Test_Data['Impurity'] != 0.0) else Test_Data['Impurity']                   
                MaxDepth = LoopParameter if Test_Data['MaxDepth'] != None else Test_Data['MaxDepth']
                MaxLeaf = LoopParameter if Test_Data['MaxLeaf'] != None else Test_Data['MaxLeaf']
                MinSampLeaf = LoopParameter if Test_Data['MinSampLeaf'] != 1 else Test_Data['MinSampLeaf']
                MinSampSplit = LoopParameter if Test_Data['MinSampSplit'] != 2 else Test_Data['MinSampSplit']

                name = path+name
                #Creates Gini Models
                pred_acc= CreateModel(Criterion=Crit,Impurity=Impurity,MaxDepth=MaxDepth,MaxLeaf=MaxLeaf,
                                                    MinSampLeaf=MinSampLeaf,MinSampSplit=MinSampSplit,Test_Data=Test_Data, X=X,
                                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                    name=name, attackNames=attackNames)
                if (Crit == 'gini') :
                    acc_gini.append(pred_acc)
                else:
                    acc_entropy.append(pred_acc)
            x_label.append(LoopParameter)
            print("Loop: "+str(LoopParameter))
        #Plots Accuracies in a Graph
        plottingGraphs(x_label=x_label,acc_gini=acc_gini,acc_entropy=acc_entropy,Test_Data=Test_Data,ImpureLoop=0,dataName=dataName)

# plots gini accuracies vs entropy accuracies graph, Saves plot diagram in Dataset Folder
def plottingGraphs(x_label,acc_gini,acc_entropy,Test_Data,ImpureLoop,dataName):
    #stores the both gini and entropy accuraccies into a dataframe to use to plot the accuracies
    d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
            'acc_entropy':pd.Series(acc_entropy),
            'x_label':pd.Series(x_label)})

    #TempNames are made due to appending of Names, This is used to rename to original value
    TempName = Test_Data['Name']
    TempName2 = Test_Data['Table_Name']
    fig = plt.figure()
    plt.plot('x_label','acc_gini', data=d, label='gini')
    plt.plot('x_label','acc_entropy', data=d, label='entropy')
    plt.xlabel(Test_Data['Table_Name'])
    plt.ylabel('accuracy')

    #Changes Path Based on Parameters
    if(Test_Data['TwoParameters']):
        Test_Data['Table_Name'] = Test_Data['Table_Name']+' Impurity('+str(ImpureLoop)+')'
        Test_Data['Name'] = Test_Data['Name']+'Impurity('+str(ImpureLoop)+')'
        if(Test_Data['MaxDepth'] != None):
            figPath=originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Max_Depth/"
        elif(Test_Data['MinSampLeaf'] != 1):
            figPath=originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Min_Samples_Leaf/"
        elif(Test_Data['MinSampSplit'] != 2):
            figPath=originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Min_Samples_Split/"
        else:
            figPath=originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracy_Graphs/Impurity_Decreased_Plus_Max_Leaf_Nodes/"
    else:
        figPath=originalPath+"IDS_SeniorProject2020/DecisionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracy_Graphs/Gini_vs_Entropy_Accuracy_Graphs/"

    #Set up Plot Diagram and Saves it        
    plt.title('Gini vs Entropy - Accuracy vs '+Test_Data['Table_Name'])
    plt.legend()
    #plt.show()
    figPath = figPath+'Gini_vs_Entropy_Feature_'+Test_Data['Name']
    fig.savefig(figPath+'.png')
    Test_Data['Name']=TempName
    Test_Data['Table_Name']=TempName2
