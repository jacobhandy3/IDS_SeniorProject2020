import NNcode
import decisionTree
import load_Dataset
import numpy
import pandas 

#Loads CIC dataset
def Load_First_DataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum=load_Dataset.firstDataset()
    dataset = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL)
    return dataset,Xmax,attackNum,labelCol

#Loads UNSW-NB15 dataset
def Load_SecondDataSet():
    path,header,indexCol,mapped,colL,Xmax,labelCol,attackNum,dropFeats,missReplacement,missCols=load_Dataset.secondDataset()
    dataset = load_Dataset.loadDataset(path=path,header=header,indexCol=indexCol,labelCol=labelCol,mapped=mapped,colL=colL,
                                            dropFeats=dropFeats,missReplacement=missReplacement,missCols=missCols)
    return dataset,Xmax,attackNum,labelCol

#Runs the NN File
def Run_NNcode(dataset,Xmax,attackNum,labelCol):
    X,model = NNcode.NNanalysis(dataset,Xmax=Xmax,labelCol=labelCol,attackNum=attackNum)
    return X,model

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



#Loaded 1st Dataset
dataset,Xmax,attackNum,labelCol=Load_First_DataSet()
#Ran Decision Tree
Run_DecisionTree(dataset=dataset,Xmax=Xmax,attackNum=attackNum,labelCol=labelCol)

#I Used this to Check any errors I had
def testCode(inputData,target,dataset):    
    #print("Input Data: ")
    #print(inputData)
    #print("Target: ")
    print(target)
    attacks = target[:,].unique()
    print(sorted(attacks))

    #inputData = inputData.drop(dataset.columns[14],axis=1)
    #inputData = inputData.drop(dataset.columns[15],axis=1)

    print(inputData.info())
    print(dataset.info())
'''
print("PRINTING MODEL 2")
cnt = 31
for x in range(cnt):
    rowNum =X.iloc[[x]] 
    print("Target: ",target.iloc[[x]])
    decisionTree.predictM(rowNum=rowNum, model=model)
print("PRINTING MODEL 4")
for x in range(cnt):
    rowNum =X.iloc[[x]] 
    print("Target: ",target.iloc[[x]])
    decisionTree.predictM(rowNum=rowNum, model=model4)
print("PRINTING MODEL 6")
for x in range(cnt):
    rowNum =X.iloc[[x]] 
    print("Target: ",target.iloc[[x]])
    decisionTree.predictM(rowNum=rowNum, model=model6)
print("PRINTING MODEL 8")
for x in range(cnt):
    rowNum =X.iloc[[x]] 
    print("Target: ",target.iloc[[x]])
    decisionTree.predictM(rowNum=rowNum, model=model8)
'''
'''
X,model = decisionTree.fineTune(X=X, model=model,attackNum=attackNum,Xmax=Xmax,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)

for x in range(21):
    rowNum =X.iloc[[x]] 
    print("Target: ",target.iloc[[x]])
    decisionTree.predictM(rowNum=rowNum, model=model)
'''