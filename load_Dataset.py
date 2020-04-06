#Import Libraries
import pandas as pd
import numpy as np
import glob as glob
import os
from sklearn.preprocessing import LabelEncoder #ADDED THIS IMPORT TO ENCODE STRING TO INT

#Contains all the Attributes for the CIC Dataset
def firstDataset():
    #Dictionary of all the specific attacksCIC for CICIDS Dataset
    attacksCIC = {
    "BENIGN": 0,
    "FTP-Patator": 1,
    "SSH-Patator": 2,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 4,
    "DoS Hulk": 5,
    "DoS GoldenEye": 6,
    "Heartbleed": 7,
    "Web Attack Brute Force": 8,
    "Web Attack XSS": 9,
    "Web Attack Sql Injection": 10,
    "Infiltration": 11,
    "Bot": 12,
    "PortScan": 13,
    "DDoS": 14,
    }
    #Rows to review CIC data and add to dictionary
    CICrows = [78]
    
    return r"/media/southpark86/AMG1/School/Spring 2020/Senior Project/MachineLearningCVE", 0, None, attacksCIC, CICrows, 77, 78, 15

#Contains all the Attributes for the UNSW Dataset
def secondDataset():

    #Dictionary of all specific attacksCIC for 2nd dataset
    mappingUNSW = {
    #attacks
    "Benign": 0,
    " Fuzzers": 1,
    "Reconnaissance": 2,
    "Shellcode": 3,
    "Analysis": 4,
    "Backdoor": 5,
    "DoS": 6,
    "Exploits": 7,
    "Generic": 8,
    "Worms": 9,
    }
    #Columns to review UNSW data and add to dictionary
    UNSWcols = [0,1,2,3,11,45]

    return r"/media/southpark86/AMG1/School/Spring 2020/Senior Project/UNSW-NB15", None, None, mappingUNSW, UNSWcols, 44, 45, 10, [1,3], ["Benign"], [47]

def loadDataset(path, header, indexCol, colL,labelCol, mapped, dropFeats=[], missReplacement=[], missCols=[]):
    print("Pre-processing data...")
    #get formatted pandas dataset
    dataset = formatData(path, header, indexCol)

    #Drop columns not using
    if(len(dropFeats) != 0):
        dataset.drop(dropFeats, axis=1, inplace=True)
    #Fill missing data with columns with missing data
    if(len(missReplacement) != 0):
        dataset = missData(dataset, missReplacement, missCols)

    #replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    dataset = dataset.replace(to_replace=-np.inf, value=np.nan)
    #Added This Replace To Help with CIC Dataset
    dataset = dataset.replace(to_replace="Infinity", value=np.nan)

    #drop any row with a NaN
    dataset = dataset.dropna(how="any")
    #Replaces attacks with numbers
    for columnL in colL:
        encodeData = LabelEncoder()
        dataset.iloc[:,columnL] = encodeData.fit_transform(dataset.iloc[:,columnL]) #ADDED THIS CODE; AUTO GENERATES A NUMBERS FOR EACH STRING
    
    print("done")
    return dataset

#takes existing map, pandas data frame, and list of columns to review
def AddToMap(m, ds, cL):
    i = 0
    #for each column in list
    for c in cL:
        #for each cell in the column
        for cell in ds.iloc[:,c]:
            #if already in map, do nothing
            if cell in m:
                continue
            #if not in map, add to map with i
            else:
                m[cell] = i
                i+=1
        #reset counter for next column
        i = 0
    #return the map
    return m

#takes a folde path to find csv files, 0 for a header and None for no header
#and 0 for an index column or None for no index column
def formatData(path, head, indexCol):
    #open dataset folder path
    os.chdir(path)
    #find files with glob
    fileList = glob.glob("*.csv")
    #create a temp list
    dataList = []
    #loop thorugh the files
    for file in fileList:
        #read each file as csv with pandas
        data = pd.read_csv(file, header=head, index_col=indexCol)
        #append to temp list
        dataList.append(data)
    #concat vertically
    dataset = pd.concat(dataList, axis=0)
    #return the dataset
    return dataset

#takes a dataset, list of value to replace missing data in
#corresponding column in missCols
def missData(dataset, miss, missCols):
    i = 0
    #for each column in list of columns, missCols
    for col in missCols:
        #With the column, fill any missing data with item in miss list
        dataset[col].fillna(value=miss[i], inplace=True)
        #increment counter for next item to use for next column
        i+=1
    #return the dataset
    return dataset
    i = 0
    #for each column in list of columns, missCols
    for col in missCols:
        #With the column, fill any missing data with item in miss list
        dataset[col].fillna(value=miss[i], inplace=True)
        #increment counter for next item to use for next column
        i+=1
    #return the dataset
    return dataset
