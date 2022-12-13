import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU,Dense,Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# load data
data = pd.read_csv("PowerVsWeater.csv")

data = np.asarray(data)

features = data[:,1:9]

numFeatures = 8

powerOutputs = data[:,9:]
print(features.shape)

# split data to test and train set
fTrain,fTest,pTrain,pTest = train_test_split(features, powerOutputs, test_size=.2)
print(pTrain.shape)
numSamples = pTrain.size

# initialize our model. Input layer, followed by GRU, followed by an Output later
model = Sequential()
model.add(Input(shape=(fTrain.shape[1],1)))
model.add(GRU(units=32,input_shape=(1,8)))

model.add(Dense(units=1))

# train model on training data
model.compile(loss='mse',optimizer='adam')
model.fit(fTrain,pTrain,epochs=300)

# create predictions
predictions = model.predict(fTest)
for i in range(predictions.size):
    if predictions[i] < 0.0:
        predictions[i] = 0.0

# compare predictions in a plot
plt.plot(predictions,c='b')
plt.plot(pTest,c='g')
plt.show()
