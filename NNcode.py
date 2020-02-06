from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('DataSets\CIC-IDS-2017\Tuesday-WorkingHours.pcap_ISCX (C).csv', delimiter=',')


# split into input (X) and output (y) variables
X = dataset[:,:78]
y = dataset[:,78]


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=78, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)


# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))