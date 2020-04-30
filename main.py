import NNcode
import decisionTree
import load_Dataset
import numpy
import pandas as pd
#Criteria for each Testing Parameters
Criterion = ['entropy', 'gini']
Test0 = {'MaxDepth': range(1,31),'Impurity': 0.0,'MaxLeaf': None,'MinSampSplit':2,'MinSampLeaf':1,'Name': "Max_Depth",'Table_Name': "Max_Depth",'TwoParameters': False}
Test1 = {'MaxDepth': None,'Impurity': 0.0,'MaxLeaf': range(2,31),'MinSampSplit':2,'MinSampLeaf':1,'Name': "Max_Leaf_Nodes",'Table_Name': "Max_Leaf_Nodes",'TwoParameters': False}
Test2 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005,.01,.05,.1,.5,1],'MaxLeaf': None,'MinSampSplit':2,'MinSampLeaf':1,'Name': "Impurity_Decreased",'Table_Name': "Impurity_Decreased",'TwoParameters': False}
Test3 = {'MaxDepth': range(1,31),'Impurity': [.00005,.0001,.0005,.001,.005],'MaxLeaf': None,'MinSampSplit':2,'MinSampLeaf':1,'Name': "Impurity_Decreased_Plus_Max_Depth",'Table_Name': "Max_Depth",'TwoParameters': True}
Test4 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005],'MaxLeaf': range(2,31),'MinSampSplit':2,'MinSampLeaf':1,'Name': "Impurity_Decreased_Plus_Max_Leaf_Nodes",'Table_Name': "Max_Leaf_Nodes",'TwoParameters': True}

Test5 = {'MaxDepth': None,'Impurity': 0.0,'MaxLeaf': None,'MinSampSplit':2,'MinSampLeaf':range(50,1501,50),'Name': "Min_Samples_Leaf",'Table_Name': "Min_Samples_Leaf",'TwoParameters': False}
Test6 = {'MaxDepth': None,'Impurity': 0.0,'MaxLeaf': None,'MinSampSplit':range(50,1501,50),'MinSampLeaf':1,'Name': "Min_Samples_Split",'Table_Name': "Min_Samples_Split",'TwoParameters': False}
Test7 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005],'MaxLeaf': None,'MinSampSplit':2,'MinSampLeaf':range(50,1501,50),'Name': "Impurity_Decreased_Plus_Min_Samples_Leaf",'Table_Name': "Min_Samples_Leaf",'TwoParameters': True}
Test8 = {'MaxDepth': None,'Impurity': [.00005,.0001,.0005,.001,.005],'MaxLeaf': None,'MinSampSplit':range(50,1501,50),'MinSampLeaf':1,'Name': "Impurity_Decreased_Plus_Min_Samples_Split",'Table_Name': "Min_Samples_Split",'TwoParameters': True}

Test_List = [Test0,Test1,Test2,Test3,Test4,Test5,Test6,Test7,Test8]

filePath=r"/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/"

#Saves CIC dataset
def SaveDataSet1():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/firstDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/firstDataSetAttackNames.csv")

#Saves UNSW-NB15 dataset
def SaveDataSet2():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.secondDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL,
                                            dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)    
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
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.fourthDataset()
    dataset,attackNames = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    dataset.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/fourthDataSet.csv")
    attackNames.to_csv("/media/southpark86/AMG1/School/Spring 2020/Senior Project/DataSets/fourthDataSetAttackNames.csv")

#Loads CIDDS dataset
def Load_ThirdDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.thirdDataset()
    
    #Loading Saved CSV    
    fileP = filePath+"Third_Data_Set/thirdDataSet.csv"
    dataset = load_Dataset.load_load(path=fileP,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
    

    fileP = filePath+"Third_Data_Set/thirdDataSetAttackNames.csv"
    attackNames = pd.read_csv(fileP, header=header, index_col=indexCol)
    attackNames = attackNames.iloc[:,1]
    attackNames = attackNames.unique()
    print(dataset)
    return dataset,Xmax,attackNum,labelCol,attackNames


#SaveDataSet1()
#SaveDataSet2()
#SaveDataSet3()
#SaveDataSet4()

def Run_Test():
    #Load Datasets
    for LoopNum in range(1-5):
        dataset,attackNames,labelCol=load_Dataset.Load_Saved_Data_Set(LoopNum)
        data_name = "UNSW" if (LoopNum == 2) else ("DDOS16" if (LoopNum == 4) else ("CICIDS" if (LoopNum == 1) else "CIDDS"))
        #Run Decision Tree
        for Test_Num in Test_List:
            decisionTree.Run_DecisionTree(dataset=dataset,attackNames=attackNames,labelCol=labelCol,Criterion=Criterion,dataName=data_name,Test_Data=Test_Num)
        print("FINISHED DATASET: "+str(LoopNum))

Run_Test()


#Loaded 3nd Dataset
#dataset,Xmax,attackNum,labelCol,attackNames=Load_SecondDataSet()
#Ran Decision Tree
#Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol,attackNames=attackNames,dataName="CIDDS")
#print("FINISHED SECOND DATASET")
