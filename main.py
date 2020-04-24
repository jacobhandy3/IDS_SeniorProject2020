import NNcode
import decisionTree
import load_Dataset
import numpy
import pandas as pd
import matplotlib.pyplot as plt
#Loads CIC dataset
def Load_First_DataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    return dataset,Xmax,attackNum,labelCol,attackNames

#Loads UNSW-NB15 dataset
def Load_SecondDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.secondDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL,
                                            dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
    return dataset,Xmax,attackNum,labelCol, attackNames

#Runs the NN File
def Run_NNcode(dataset,Xmax,attackNum,labelCol):
    X,model = NNcode.NNanalysis(dataset,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
    return X,model

#Runs the Decision Tree File
def Run_DecisionTree(dataset,Xmax,attackNum,labelCol,attackNames):
    attackNames = attackNames.unique()

    #Split and Train Data
    X,y,X_train,y_train,X_test,y_test= decisionTree.trainModels(dataset,labelCol)

    #plot gini vs entropy accuracies
    max_depth = []
    acc_gini = []
    acc_entropy = []

    #Create and Printing Models, Appends predicted accuracies based on Criterion, Appends MaxDepth
    for MaxDepth in range(2,30):
        path="/media/southpark86/AMG1/School/Spring 2020/Senior Project/DecionTreeResults/CICIDS Dataset/Criterion_Gini/Splitter_Plus_treeDepth/Splitter_Best/"
        Criterion='gini'
        name = 'CIC_Dataset_Features_Criterion_Gini_Splitter_Random_TreeDepth_'+str(MaxDepth)+'.png'
        name = path+name
        #Creates Gini Models and Prints MaxDepth < 10
        pred_acc= decisionTree.CreateModel(MaxDepth=MaxDepth,Criterion=Criterion,X=X,
                                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                    name=name, attackNames=attackNames)
        acc_gini.append(pred_acc)

        path="/media/southpark86/AMG1/School/Spring 2020/Senior Project/DecionTreeResults/CICIDS Dataset/Criterion_Entropy/Splitter_Plus_treeDepth/Splitter_Best/"
        Criterion='entropy'
        name = 'CIC_Dataset_Features_Criterion_Entropy_Splitter_Random_TreeDepth_'+str(MaxDepth)+'.png'
        name = path+name
        #Creates Entropy Models and Prints MaxDepth < 10
        pred_acc= decisionTree.CreateModel(MaxDepth=MaxDepth,Criterion=Criterion,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name, attackNames=attackNames)
        
        acc_entropy.append(pred_acc)
        max_depth.append(MaxDepth)
        print()

    #stores the both gini and entropy accuraccies into a dataframe to use to plot the accuracies
    d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
            'acc_entropy':pd.Series(acc_entropy),
            'max_depth':pd.Series(max_depth)})
    # plots gini accuracies vs entropy accuracies graph, Saves plot diagram in Dataset Folder
    fig = plt.figure()
    plt.plot('max_depth','acc_gini', data=d, label='gini')
    plt.plot('max_depth','acc_entropy', data=d, label='entropy')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.title('Gini vs Entropy')
    plt.legend()
    plt.show()
    fig.savefig('gini_vs_entropy.png')



#Loaded 1st Dataset
dataset,Xmax,attackNum,labelCol,attackNames=Load_First_DataSet()
#Ran Decision Tree
Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames)