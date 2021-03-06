# Import Libraries
import pandas as pd
import numpy as np
import glob as glob
import os
# ADDED THIS IMPORT TO ENCODE STRING TO INT
from sklearn.preprocessing import LabelEncoder


def firstDataset():
    # Dictionary of all the specific attacksCIC for CICIDS Dataset
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
    # Rows to review CIC data and add to dictionary
    CICrows = [78]

    return r"C:\Users\jakem\Documents\GitHub\2020\IDS_SeniorProject2020\DataSets\CIC-IDS-2017", 0, None, attacksCIC, CICrows, 77, 78, 15


def secondDataset():

    # Dictionary of all specific attacksCIC for 2nd dataset
    mappingUNSW = {
        # attacks
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
    # Columns to review UNSW data and add to dictionary
    UNSWcols = [0, 1, 2, 3, 11, 45]

    return r"C:\Users\jakem\Documents\GitHub\2020\IDS_SeniorProject2020\DataSets\UNSW-NB15", None, None, mappingUNSW, UNSWcols, 44, 45, 10, [1, 3], ["Benign"], [47]


def thirdDataset():

    # Dictionary of all specific attacksCIC for 3rd dataset
    mappingCIDDS = {
        # attacks
        "portScan": 1,
        "dos": 2,
        "pingScan": 3,
        "bruteForce": 4
    }
    # Columns to review CIDDS data and add to dictionary
    CIDDScols = [2, 10, 13]

    return r"C:\Users\jakem\Documents\GitHub\2020\IDS_SeniorProject2020\DataSets\CIDDS-01", 0, None, mappingCIDDS, CIDDScols, 10, 13, 4, [0, 3, 5, 9, 12, 14, 15], ["Benign"], [13]


def fourthDataset():

    # Dictionary of all specific attacksCIC for 4th dataset
    mappingDDOS = {
        # attacks
        "Normal": 0,
        "HTTP-FLOOD": 1,
        "SIDDOS": 2,
        "Smurf": 3,
        "UDP-Flood": 4
    }
    # Columns to review DDOS16 data and add to dictionary
    DDOScols = [5, 7, 12, 13, 27]

    return r"C:\Users\jakem\Documents\GitHub\2020\IDS_SeniorProject2020\DataSets\DDOS16", 0, None, mappingDDOS, DDOScols, 26, 27, 5


def loadDataset(path, header, indexCol, colL, labelCol, mapped, dropFeats=[], missReplacement=[], missCols=[]):
    print("Pre-processing data...")
    # get formatted pandas dataset
    dataset = formatData(path, header, indexCol)

    # Drop columns not using
    if(len(dropFeats) != 0):
        dataset.drop(dropFeats, axis=1, inplace=True)
    # Fill missing data with columns with missing data
    if(len(missReplacement) != 0):
        dataset = missData(dataset, missReplacement, missCols)

    # replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    # drop any row with a NaN
    dataset = dataset.dropna(how="any")
    # Replaces attacks with numbers
    for columnL in colL:
        encodeData = LabelEncoder()
        dataset.iloc[:, columnL] = encodeData.fit_transform(
            dataset.iloc[:, columnL])

    print("done")
    return dataset


def loadThirdDataset(path, header, indexCol, colL, labelCol, mapped, dropFeats=[], missReplacement=[], missCols=[]):
    # get formatted pandas dataset
    dataset = loadDataset(path, header, indexCol, colL, labelCol,
                          mapped, dropFeats=[], missReplacement=[], missCols=[])
    print("Dealing with troublesome column...")
    # column that abbreviates large byte sizes with an "M"
    byteCol = 8
    temp = ''
    num = 0
    # loop through each cell, detect "M", delete it, replace with actual value
    for cell in range(dataset.shape[0]-1):
        temp = str(dataset.iloc[cell, byteCol])
        if(temp.find("M") != -1):
            #print("replaced " + temp + ": " + str(cell))
            tempN = temp.replace(" M", '')
            num = float(tempN) * 1048576
            dataset["Bytes"].replace(to_replace=temp, value=num, inplace=True)
    dataset["Bytes"] = dataset["Bytes"].astype(float)
    return dataset


# takes existing map, pandas data frame, and list of columns to review
def AddToMap(m, ds, cL):
    i = 0
    # for each column in list
    for c in cL:
        # for each cell in the column
        for cell in ds.iloc[:, c]:
            # if already in map, do nothing
            if cell in m:
                continue
            # if not in map, add to map with i
            else:
                m[cell] = i
                i += 1
        # reset counter for next column
        i = 0
    # return the map
    return m

# takes a folde path to find csv files, 0 for a header and None for no header
# and 0 for an index column or None for no index column


def formatData(path, head, indexCol):
    # open dataset folder path
    os.chdir(path)
    # find files with glob
    fileList = glob.glob("*.csv")
    # create a temp list
    dataList = []
    # loop thorugh the files
    for file in fileList:
        # read each file as csv with pandas
        data = pd.read_csv(file, header=head, index_col=indexCol)
        # append to temp list
        dataList.append(data)
    # concat vertically
    dataset = pd.concat(dataList, axis=0)
    # return the dataset
    return dataset

# takes a dataset, list of value to replace missing data in
# corresponding column in missCols


def missData(dataset, miss, missCols):
    i = 0
    # for each column in list of columns, missCols
    for col in missCols:
        # With the column, fill any missing data with item in miss list
        dataset[col].fillna(value=miss[i], inplace=True)
        # increment counter for next item to use for next column
        i += 1
    # return the dataset
    return dataset
    i = 0
    # for each column in list of columns, missCols
    for col in missCols:
        # With the column, fill any missing data with item in miss list
        dataset[col].fillna(value=miss[i], inplace=True)
        # increment counter for next item to use for next column
        i += 1
    # return the dataset
    return dataset
