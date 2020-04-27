import NNcode
import decisionTree
import load_Dataset
import numpy
import pandas as pd
import matplotlib.pyplot as plt
#Saves CIC dataset
def SaveDataSet1():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/firstDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/firstDataSetAttackNames.csv")

#Saves UNSW-NB15 dataset
def SaveDataSet2():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/secondDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/secondDataSetAttackNames.csv")

#Saves CIDDS dataset
def SaveDataSet3():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.thirdDataset()
    dataset,attackNames = load_Dataset.loadThirdDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/thirdDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/thirdDataSetAttackNames.csv")

#Saves DDOS16 dataset
def SaveDataSet4():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/fourthDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/fourthDataSetAttackNames.csv")

#Loads CIC dataset
def Load_First_DataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()

    return dataset,Xmax,attackNum,labelCol,attackNames

#Loads UNSW-NB15 dataset
def Load_SecondDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.secondDataset()

    return dataset,Xmax,attackNum,labelCol, attackNames

#Loads CIDDS dataset
def Load_ThirdDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.thirdDataset()
    
    return dataset,Xmax,attackNum,labelCol,attackNames

#Loads DDOS16 dataset
def Load_FourthDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.fourthDataset()

    return dataset,Xmax,attackNum,labelCol,attackNames

#Runs the Decision Tree File
def Run_DecisionTree(dataset,Xmax,attackNum,labelCol,attackNames,dataName):
    attackNames = attackNames.unique()

    #Split and Train Data
    X,y,X_train,y_train,X_test,y_test= decisionTree.trainModels(dataset,labelCol)

    #plot gini vs entropy accuracies
    max_depth = []
    acc_gini = []
    acc_entropy = []

    #Create and Printing Models, Appends predicted accuracies based on Criterion, Appends MaxDepth
    #Paths changed according to tested parameters
    #Max Depth change according to stopping criteria (max_depth: range(1,31), min_samples_split: range((50,1501,50)), min_samples_leaf: range(5,301,5), max_leaf_nodes)
    for MaxDepth in range(1000,30001,1000):
        path="/media/southpark86/AMG1/School/Spring 2020/Senior Project/IDS_SeniorProject2020/DecionTreeResults/"+dataName+" Dataset/Criterion_Gini/Min_Sample_Split/"
        Criterion='gini'
        name = dataName+'_Dataset_Features_Criterion_Gini_Min_Sample_Split_'+str(MaxDepth)+'.png'
        name = path+name
        #Creates Gini Models and Prints MaxDepth < 10
        pred_acc= decisionTree.CreateModel(MaxDepth=MaxDepth,Criterion=Criterion,X=X,
                                                    X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                    name=name, attackNames=attackNames)
        acc_gini.append(pred_acc)

        path="/media/southpark86/AMG1/School/Spring 2020/Senior Project/IDS_SeniorProject2020/DecionTreeResults/"+dataName+" Dataset/Criterion_Entropy/Min_Sample_Split/"
        Criterion='entropy'
        name = dataName+'_Dataset_Features_Criterion_Entropy_Min_Sample_Split_'+str(MaxDepth)+'.png'
        name = path+name
        #Creates Entropy Models and Prints MaxDepth < 10
        pred_acc= decisionTree.CreateModel(MaxDepth=MaxDepth,Criterion=Criterion,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name, attackNames=attackNames)
        
        acc_entropy.append(pred_acc)
        max_depth.append(MaxDepth)
        print("MaxDepth: ",MaxDepth)

    #stores the both gini and entropy accuraccies into a dataframe to use to plot the accuracies
    d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
            'acc_entropy':pd.Series(acc_entropy),
            'max_depth':pd.Series(max_depth)})
    # plots gini accuracies vs entropy accuracies graph, Saves plot diagram in Dataset Folder
    fig = plt.figure()
    plt.plot('max_depth','acc_gini', data=d, label='gini')
    plt.plot('max_depth','acc_entropy', data=d, label='entropy')
    plt.xlabel('min_samples_split')
    plt.ylabel('accuracy')
    plt.title('Gini vs Entropy - Accuracy vs Min_Samples_Split')
    plt.legend()
    #plt.show()
    figPath="/media/southpark86/AMG1/School/Spring 2020/Senior Project/IDS_SeniorProject2020/DecionTreeResults/"+dataName+" Dataset/Gini_vs_Entropy_Accuracies_Graphs/"
    fig.savefig(figPath+'Gini_vs_Entropy_Feature_Min_Samples_Split.png')

#SaveDataSet1()
#SaveDataSet2()
#SaveDataSet3()
#SaveDataSet4()

#Loaded 1st Dataset
#dataset,Xmax,attackNum,labelCol,attackNames=Load_First_DataSet()
#Ran Decision Tree
#Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames,dataName="CICIDS")
#print("FINISHED FIRST DATASET")

#Loaded 2nd Dataset
#dataset,Xmax,attackNum,labelCol,attackNames=Load_SecondDataSet()
#Ran Decision Tree
#Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames,dataName="UNSW")
#print("FINISHED SECOND DATASET")

#Loaded 3rd Dataset
#dataset,Xmax,attackNum,labelCol,attackNames=Load_ThirdDataSet()
#Ran Decision Tree
#Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames,dataName="CIDDS")
#print("FINISHED THIRD DATASET")

#Loaded 4th Dataset
#dataset,Xmax,attackNum,labelCol,attackNames=Load_FourthDataSet()
#Ran Decision Tree
#Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames,dataName="DDOS16")
#print("FINISHED FORTH DATASET")