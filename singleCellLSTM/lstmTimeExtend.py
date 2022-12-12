import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tensorflow as tf
from tensorflow import keras
from keras import layers
#load data
data = pd.read_csv("PowerVsWeather.csv", decimal=',')
print(data.head())


values = data.values
#get conditions and power output
conditions = values[:,1:9]
power = values[:,9]
power = np.array(power, dtype="float32")
plt.scatter(range(len(power)), power)
plt.show()

#number of past values passed in
pastSize = 672 #uses the past week of weather values
futureSize = 48 #to predict the next 12 hours of power outsputs
#normalize 
scaler = MinMaxScaler(feature_range=(0,1))
normData = scaler.fit_transform(conditions)


#creating train test indexes  
length =len(power)-pastSize-futureSize # cannot include indices smaller than our history size
randomIdx = np.random.choice(np.arange(pastSize, len(power)-futureSize), length,  replace=False)

lenTrain = int(np.size(randomIdx)*0.8)

split = np.split(randomIdx, [lenTrain, np.size(randomIdx)])
trainingIdx = split[0]
testingIdx = split[1]
x_train = []
y_train = []

for i in trainingIdx:
    x_train.append(normData[i-pastSize:i,:]) #define length of input
    y_train.append(power[i:i+futureSize])        #define test length

#turn list into np arrays dimensions
#(num test, number is history, num dimensions)
x_train = np.array(x_train, dtype=float)
y_train = np.array(y_train, dtype=float)

x_test = []
y_test = []
for i in testingIdx:
    x_test.append(normData[i-pastSize:i,:])
    y_test.append(power[i:i+futureSize])
x_test = np.array(x_test, dtype=float)
y_test = np.array(y_test,dtype=float)





#create model
model = keras.Sequential()
model.add(layers.LSTM(200, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(layers.LSTM(200, return_sequences=False)) 
model.add(layers.Dropout(.20))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dropout(.20))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(futureSize))
model.summary()
model = tf.keras.models.load_model('./lstmModel')
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 100, epochs=1)
predictions = model.predict(x_test)
predictions = np.array(predictions)
model.save('./lstmModel')


for i in range(len(predictions)):
    time = np.arange(testingIdx[i], testingIdx[i]+futureSize)
    plt.scatter(time, predictions[i], c='r')
    plt.scatter(time, y_test[i],  c='b')
    plt.show()
