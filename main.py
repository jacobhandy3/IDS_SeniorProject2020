from IPython import embed
import numpy as np 
import pickle
import re
import csv
import utils
import random
import json
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import ops
import pandas as pd
np.random.seed(1337)

#load csv with csv reader
with open('DataSets\CIC-IDS-2017\Monday-WorkingHours.pcap_ISCX.csv', 'r') as file:
    reader = csv.reader(file)
    dataList = list(reader)
#get length of the columns of csv
colLen = len(dataList[0])   #79
#get length of the rows of csv
rowLen = len(dataList)


#create 2-D list to hold csv values
csvList = [[] for x in range(rowLen)]
i = 1
j = 0
a = 0
#filling the 2-D array
while i < rowLen:
    while j < colLen - 1:
        csvList[a].append(float(dataList[i][j]))
        j += 1
    j = 0
    a += 1
    i += 1

