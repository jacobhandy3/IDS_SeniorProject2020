from NNcode import NNanalysis
#import decisionTree
from load_Dataset import loadDataset, firstDataset, secondDataset
from KNN import KNNanalysis

#############################
#      NN Analysis
#############################
#CIC dataset
#path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=firstDataset()
#dataset1 = loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
#NNanalysis(dataset=dataset1,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)


#UNSW-NB15 dataset
path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=secondDataset()
dataset2 = loadDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
NNanalysis(dataset=dataset2,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)

#############################
#      KNN Analysis
#############################
#CIC dataset
#path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=firstDataset()
#dataset1 = loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
#KNNanalysis(dataset=dataset1,Xmax=Xmax,labelCol=labelCol)

#UNSW-NB15 dataset
#path, header, indexCol, mapped, colL, Xmax, labelCol, attackNum, dropFeats, missReplacement, missCols = secondDataset()
#dataset3 = loadDataset(path=path,header=header,indexCol=indexCol,mapped=mapped,colL=colL, labelCol=labelCol,
#                                        dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
#KNNanalysis(dataset=dataset3,Xmax=Xmax,labelCol=labelCol)

#############################
#   Decision Tree Analysis
#############################
#CIC dataset
#path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=firstDataset()
#dataset1 = loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
#decisionTree.trainModels(dataset,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)