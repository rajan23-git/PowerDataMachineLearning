import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
from keras.models import load_model
import math
import matplotlib.pyplot as plt
from important_functions import partition_set,normalize_values,splineQuadratic

#This Function predicts the future Power values
#Predicts future values given 30 previous datapoints
#num_timesteps is the number of future predictions the Function gives


def ensembleFuturePredictor(input_data,ensemble_model,complete_dataset,num_timesteps):
    input = np.copy(input_data)
    num_rows = np.shape(input)[0]
    num_cols = np.shape(input)[1]
    predictions_array = np.array([])
    #min_array and max_array are the maximum and minimum of each column of the data
    min_array = [0 for iter_var in range(5)]
    max_array = [np.max(complete_dataset[:,iter_var]) for iter_var in range(5)]
    #the input data is normalized with the mimumum and maximum of each column
    #min max normalization
    for iter in np.arange(0,num_cols):
        input[:,iter] =  (input[:,iter] - min_array[iter])/(max_array[iter] - min_array[iter])
   
    input = input[:,np.arange(0,4)]
    #future power output is predicted from previous 30 datapoints
    #the ensemble LSTM model is used for prediction
    for timestep in range(num_timesteps):
        input_size = np.shape(input)[0]
        #ensemble input is the input for the ensemble LSTM
        ensemble_input = []
        ensemble_input.append(input[input_size - 30: input_size, :])
        ensemble_input = np.array(ensemble_input)
        print("predictor_input shape = ",np.shape(ensemble_input))
        #future power value predicted using Ensemble LSTM model
        #weather data is fed into Ensemble LSTM, for prediction
        future_power = ensemble_model.predict(ensemble_input)
        predictions_array = np.append(predictions_array,future_power)

        insolation_input = []
        ambient_input = []
        module_input = []
        wind_input = []
        #insolation_input contains the input for the insolation LSTM model
        insolation_input.append(input[input_size - 30: input_size, 0])
        insolation_input = np.array(insolation_input)

        #ambient_input contains the input for the ambient LSTM model
        ambient_input.append(input[input_size - 30: input_size, 1])
        ambient_input = np.array(ambient_input)

        module_input.append(input[input_size - 30: input_size, 2])
        module_input = np.array(module_input)

        wind_input.append(input[input_size - 30: input_size, 3])
        wind_input = np.array(wind_input)
        #the future values for each weather feature are predicted
        future_insolation = insolation_model.predict(insolation_input)[0,0]
        future_ambient_temperature = ambient_model.predict(ambient_input)[0,0]
        future_module_temperature = module_model.predict(module_input)[0,0]
        future_wind_velocity = wind_model.predict(wind_input)[0,0]
        print("future wind velocity shape = ", np.shape(future_wind_velocity))
        #the future weather features are added to the input data, for the next timestep
        addedRow = np.array([future_insolation,future_ambient_temperature,future_module_temperature,future_wind_velocity])
        input = np.append(input, [addedRow], axis = 0)

    #predictions array is brought back to scale from normalization
    predictions_array = predictions_array * (max_array[4] - min_array[4]) + min_array[4]
    return predictions_array

power_model = load_model("power_model.keras")
#LSTM models for different weather features, are loaded in
insolation_model = load_model("insolation_model.keras")
ambient_model = load_model("ambient_model.keras")
module_model = load_model("module_model.keras")
wind_model = load_model("wind_model.keras")

#loads in the data
data_weather_power = np.loadtxt("../power_weather_data/PowerVsWeather.csv",delimiter = ",",skiprows = 1,dtype = float)
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
#applies quadratic splining to the data
quadratic_data = splineQuadratic(data_weather_power)

normalized_linear_data = normalize_values(data_weather_power)
normalized_quadratic_data = normalize_values(quadratic_data)

#array of maximum values, for each column
linear_max_array = [np.max(data_weather_power[:,iter_var]) for iter_var in range(5)]
#array of minimum values, for each column
linear_min_array = [0 for iter_var in range(5)]

quadratic_max_array = [np.max(quadratic_data[:,iter_var]) for iter_var in range(5)]
quadratic_min_array = [0 for iter_var in range(5)]

#the predictions from the weather condition LSTMS are loaded in
weather_predictions = np.loadtxt("linear_predictions.txt",delimiter= ",",dtype = float)
#This is the Ensemble LSTM for linear splining
linear_ensemble_model = load_model("linear_LSTM.keras")

power_values = np.transpose ( [ normalized_linear_data[:,4] ] )
unused1,x_test,unused2,y_test = partition_set(weather_predictions,power_values)

#this is the ensemble model for Linear splining
#the ensemble model is used for predictions
#The predictions from the weather condition LSTMS are fed into the Ensemble LSTM with Linear Splining
linear_ensemble_predictions = linear_ensemble_model.predict(x_test)
print("y_test shape = " , y_test.shape)
print("ensemble_predictions shape = " , linear_ensemble_predictions.shape)

#this is the graph for the Ensemble LSTM with Linear Splining
#the predicted versus actual power values of the Ensemble LSTM are graphed
plt.figure(1)
plt.title("Linear Splining Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("timestep")
plt.ylabel("power value")
length = y_test.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,y_test,label = "Actual")
plt.plot(timesteps,linear_ensemble_predictions,label = "Predicted")
plt.legend(loc = 'upper center')


weather_predictions = np.loadtxt("quadratic_predictions.txt",delimiter= ",",dtype = float)
#this is the ensemble LSTM for Quadratic Splining
quadratic_ensemble_model = load_model("quadratic_LSTM.keras")
power_values = np.transpose ( [ normalized_quadratic_data[:,4] ] )
unused1,x_test,unused2,y_test = partition_set(weather_predictions,power_values)

# predictions from weather condition LSTMS are fed into Ensemble LSTM with Quadratic Splining
quadratic_ensemble_predictions = quadratic_ensemble_model.predict(x_test)
print("y_test shape = " , y_test.shape)
print("ensemble_predictions shape = " , quadratic_ensemble_predictions.shape)

#The Predicted versus Actual Power Values are graphed for Ensemble LSTM with Quadratic Splining
plt.figure(2)
plt.title("Quadratic Splining Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("timestep")
plt.ylabel("power value")
length = y_test.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,y_test,label = "Actual")
plt.plot(timesteps,quadratic_ensemble_predictions,label = "Predicted")
plt.legend(loc = 'upper center')

#This generates input data that increases at a constant rate, using linspace
#This input data is used to test the ensembleFuturePredictor Function
input_data_linear = np.ones((30,1))
for i in np.arange(0,5):
    #data is generated within the minimum and maximum value of each weather feature
    #and power
    created_column_data = np.linspace(linear_min_array[i], linear_max_array[i],30)
    created_column_data = np.transpose([created_column_data])
    input_data_linear = np.append(input_data_linear,created_column_data,axis = 1)

input_data_quadratic = np.ones((30,1))
for i in np.arange(0,5):
    
    created_column_data = np.linspace(quadratic_min_array[i], quadratic_max_array[i],30)
    created_column_data = np.transpose([created_column_data])
    input_data_quadratic = np.append(input_data_quadratic,created_column_data,axis = 1)
    
input_data_linear = np.delete(input_data_linear,0, axis = 1)
input_data_quadratic = np.delete(input_data_quadratic,0, axis = 1)
    

#future predictions for the Linear Splining LSTM
power_prediction_linear = ensembleFuturePredictor(input_data_linear,linear_ensemble_model,data_weather_power,20)
#future predictions for the Quadratic Splining LSTM
power_prediction_quadratic = ensembleFuturePredictor(input_data_quadratic,quadratic_ensemble_model,quadratic_data,20)

print("power_prediction_linear = ", power_prediction_linear)
print("power_prediction_quadratic = ", power_prediction_quadratic)

#Graphs the output of the ensembleFuturePredictor function
#plots future power predictions for LSTM with Linear splining
plt.figure(3)
plt.title("Future Power Predictions from Ensemble LSTM with Linear Splining")
plt.xlabel("timestep")
plt.ylabel("power values")
num_prev_points = np.shape(input_data_linear)[0]
plt.scatter(np.arange(0,num_prev_points) , input_data_linear[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_linear)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_linear,label = "Future Predictions" )
plt.legend(loc = "upper center")

#Graphs the output of the ensembleFuturePredictor function
#plots the future power predictions for LSTM with Quadratic Splining
plt.figure(4)
plt.title("Future Power Predictions from Ensemble LSTM with Quadratic Splining")
plt.xlabel("timestep")
plt.ylabel("power values")
num_prev_points = np.shape(input_data_quadratic)[0]
plt.scatter(np.arange(0,num_prev_points) , input_data_quadratic[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_quadratic)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_quadratic,label = "Future Predictions" )
plt.legend(loc = "upper center")

plt.show()
print("Linear ensemble predictions = " , linear_ensemble_predictions)
print("Quadratic ensemble predictions = ", quadratic_ensemble_predictions)








