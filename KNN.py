import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_Dataset import loadDataset, firstDataset, secondDataset

def KNNanalysis(dataset, Xmax, labelCol):

    X,y = formatData(dataset, Xmax, labelCol)

    #split data with 0.32 test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)
    print("Now onto the ML code")

    feature_number = Xmax

    k = 5

    x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
    y_data_train = tf.placeholder(shape=[None, len(y[0])], dtype=tf.float32)
    x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

    # manhattan distance
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

    # nearest k points
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_label = tf.gather(y_data_train, top_k_indices)

    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)

    # nearest k points
    _, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    top_k_label = tf.gather(y_data_train, top_k_indices)

    sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_up_predictions, axis=1)

    sess = tf.Session()
    prediction_outcome = sess.run(prediction, feed_dict={x_data_train: X_train,
                                x_data_test: X_test,
                                y_data_train: y_train})

    # evaluation
    accuracy = 0
    for pred, actual in zip(prediction_outcome, y_test):
        if pred == np.argmax(actual):
            accuracy += 1

    print(accuracy / len(prediction_outcome))

def formatData(dataset, Xmax, labelCol):
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = dataset.iloc[:,0:Xmax]
    y = dataset.iloc[:,labelCol]

    # one hot encoding
    y = np.eye(len(set(y)))[y]

    return X,y