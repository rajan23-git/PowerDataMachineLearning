import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
import math
import matplotlib.pyplot as plt
def normalize_values(data_matrix):
    for iter in np.arange(0,np.shape(data_matrix)[1]):    
        data_matrix[:,iter] = (data_matrix[:,iter] - np.max(data_matrix[:,iter]))/(np.max(data_matrix[:,iter]) - np.min(data_matrix[:,iter]))
    return data_matrix

data_weather_power = np.loadtxt("../power_weather_data/PowerVsWeather.csv",delimiter = ",",skiprows = 1,dtype = float)
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
data_weather_power = normalize_values(data_weather_power)
# print(data_weather_power)
# np.savetxt("test_format.csv",data_weather_power,delimiter=",",fmt= "%1.4f %1.4f %1.4f %1.4f %1.4f")
insolation_values =   np.transpose( [ data_weather_power[:,0] ] ) 
ambient_temperature =  np.transpose( [ data_weather_power[:,1] ] ) 
module_temperature =   np.transpose( [ data_weather_power[:,2] ] )
wind_velocity = np.transpose( [ data_weather_power[:,3] ] )
power_values = np.transpose ( [ data_weather_power[:,4] ] )



def universal_LSTM(x_data,y_data,fig_number):
    num_rows = np.shape(x_data)[0]
    training_split = math.floor( num_rows * 0.80 )
    x_train = []
    y_train =   y_data[np.arange(30,training_split),:]
    
    x_test = []
    y_test =  y_data[np.arange(training_split - 30,num_rows),:] 

    for iterator in np.arange(30,training_split):
        appended_row = x_data[(iterator-30):iterator, :]
        x_train.append(appended_row)
    
    for iter in np.arange(training_split - 30,num_rows):
        appended_row = x_data[iter-30 : iter, :]
        x_test.append(appended_row)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2]))
    x_test = np.reshape(x_test, (np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2]))

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = keras.Sequential()
    model.add(layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2] )))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(30))
    model.add(layers.Dense(np.shape(y_data)[1]))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics = ['accuracy'])
    num_epochs = 5
    var_model = model.fit(x_train, y_train, batch_size= 400, epochs=num_epochs )

    plt.figure(fig_number)
    plt.xlabel("Epoch Number")
    plt.ylabel("Error Over Epochs")
    plt.scatter(np.arange(0,num_epochs),var_model.history['loss'])
    plt.plot(np.arange(0,num_epochs),var_model.history['loss'])
    
    var_predictions = model.predict(x_test)
    print("single var prediction = " ,var_predictions)
    print("single var predictions shape = ",np.shape(var_predictions ))
    return var_predictions




insolation_predict = universal_LSTM(insolation_values,insolation_values,1)
ambient_predict = universal_LSTM(ambient_temperature,ambient_temperature,2)
module_predict = universal_LSTM(module_temperature,module_temperature,3)
wind_predict = universal_LSTM(wind_velocity,wind_velocity,4)
combined_predictions = np.append(insolation_predict,ambient_predict,axis = 1)
combined_predictions = np.append(combined_predictions,module_predict,axis = 1)
combined_predictions = np.append(combined_predictions,wind_predict,axis = 1 )
print("combined predictions shape = ", np.shape(combined_predictions))
ensemble_predictions = universal_LSTM(combined_predictions,power_values,5)

plt.show()


