from NNcode import NNanalysis
import decisionTree
from load_Dataset import loadDataset, loadThirdDataset, firstDataset, secondDataset, thirdDataset, fourthDataset
from KNN import KNNanalysis

#############################
#      CICIDS2017
#############################
#CICIDS dataset
path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=firstDataset()
dataset1 = loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
NNanalysis(path=path,dataset=dataset1,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
KNNanalysis(dataset=dataset1,Xmax=Xmax,labelCol=labelCol)
#decisionTree.trainModels(dataset,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)

#############################
#      UNSW-NB15
#############################

#UNSW-NB15 dataset
path, header, indexCol, mapped, colL, Xmax, labelCol, attackNum, dropFeats, missReplacement, missCols = secondDataset()
dataset2 = loadDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
NNanalysis(path=path,dataset=dataset2,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
KNNanalysis(dataset=dataset2,Xmax=Xmax,labelCol=labelCol)
#decisionTree.trainModels(dataset2,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)

#############################
#        CIDDS-001
#############################

#CIDDS-01 dataset
#path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=thirdDataset()
#dataset3 = loadThirdDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
#                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
#NNanalysis(path=path,dataset=dataset3,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
#KNNanalysis(dataset=dataset3,Xmax=Xmax,labelCol=labelCol)
#decisionTree.trainModels(dataset3,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)

#############################
#        DDOS16
#############################

#DDOS16 dataset
path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=fourthDataset()
dataset4 = loadDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol)
NNanalysis(path=path,dataset=dataset4,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
KNNanalysis(dataset=dataset4,Xmax=Xmax,labelCol=labelCol)
#Run_DecisionTree(dataset=dataset4,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol)

#Runs the Decision Tree File
def Run_DecisionTree(dataset,Xmax,attackNum,labelCol):
    #Splits Dataset
    inputData,target = decisionTree.trainDataset(dataset=dataset,Xmax=Xmax,labelCol=labelCol)
    #Splits Dataset into train and test sets
    X, y,attackNum,Xmax,X_train,y_train,X_test,y_test = decisionTree.trainModels(dataset,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum,inputData=inputData,target=target)

    #The Range determines the Tree Depth
    for xp in range(2,10):
        #Name is meant to Name the PNG file
        name = 'tree'+str(xp)+'.png'
        #Creates and Runs Models
        decisionTree.CreateModel(cntr=xp,X=X,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name)
        print()