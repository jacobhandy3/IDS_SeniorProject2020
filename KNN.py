import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from math import sqrt

def KNNanalysis(dataset, Xmax, labelCol):
    tf.compat.v1.disable_eager_execution()

    X,y = formatData(dataset, Xmax, labelCol)
    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")

    n = X_train.shape[0]
    print(str(round(sqrt(n/2))))
    knnMD = KNeighborsClassifier(n_neighbors=round(sqrt(n)/2), algorithm="kd_tree",metric="manhattan")
    knnMD.fit(X_train, y_train)
    y_predBT = knnMD.predict(X_test)
    a2 = metrics.accuracy_score(y_test, y_predBT)
    print("Accuracy: " + str(a2))



#split the data from the labels
def formatData(dataset, Xmax, labelCol):
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = dataset.iloc[:,0:Xmax]
    y = dataset.iloc[:,labelCol]

    # one hot encoding
    #y = np.eye(len(set(y)))[y]

    return X,y