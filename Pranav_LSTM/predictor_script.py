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
timestamped_data = data_weather_power
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
#applies quadratic splining to the data
quadratic_data = splineQuadratic(data_weather_power)

normalized_linear_data = normalize_values(data_weather_power)
normalized_quadratic_data = normalize_values(quadratic_data)


#the predictions from the weather condition LSTMS are loaded in
weather_predictions = np.loadtxt("linear_predictions.txt",delimiter= ",",dtype = float)
#This is the Ensemble LSTM for linear splining
linear_ensemble_model = load_model("linear_LSTM.keras")

power_values = np.transpose ( [ weather_predictions[:,4] ] )
unused1,x_test,unused2,actual_values = partition_set(weather_predictions[:,0:4],power_values)

#this is the ensemble model for Linear splining
#the ensemble model is used for predictions
#The predictions from the weather condition LSTMS are fed into the Ensemble LSTM with Linear Splining
linear_ensemble_predictions = linear_ensemble_model.predict(x_test)
print("y_test shape = " , actual_values.shape)
print("ensemble_predictions shape = " , linear_ensemble_predictions.shape)

#this gets the date time range of the plotted data
#the last 4 percent of the original data is plotted
#this is because train test split was performed two times
cols = np.shape(timestamped_data)[1]
rows =  np.shape(timestamped_data)[0]
num_points = int(rows* 0.04)
date_array = timestamped_data[:,cols-5:cols-1]
starting_date  = date_array[rows - num_points - 1,:]
ending_date = date_array[rows - 1,:]
starting_date_string = str(int(starting_date[0]))+ "/"+str(int(starting_date[1]))
ending_date_string = str(int(ending_date[0]))+ "/"+str(int(ending_date[1]))
print("starting date = " ,starting_date)
print("ending date = ", ending_date)


#this is the graph for the Ensemble LSTM with Linear Splining
#the predicted versus actual power values of the Ensemble LSTM are graphed
plt.figure(1)
plt.title("Linear Splining Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("15 Minute Timesteps")
plt.ylabel("Power Value")
length = actual_values.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,actual_values,label = "Actual")
plt.plot(timesteps,linear_ensemble_predictions,label = "Predicted")
plt.text(.5, .5, 'Date Range = ' + str(starting_date_string) + " to " + str(ending_date_string), fontsize = 12)
plt.legend(loc = 'upper center')


weather_predictions = np.loadtxt("quadratic_predictions.txt",delimiter= ",",dtype = float)
#this is the ensemble LSTM for Quadratic Splining
quadratic_ensemble_model = load_model("quadratic_LSTM.keras")
power_values = np.transpose ( [ weather_predictions[:,4] ] )
unused1,x_test,unused2,actual_values = partition_set(weather_predictions[:,0:4],power_values)

# predictions from weather condition LSTMS are fed into Ensemble LSTM with Quadratic Splining
quadratic_ensemble_predictions = quadratic_ensemble_model.predict(x_test)
print("y_test shape = " , actual_values.shape)
print("ensemble_predictions shape = " , quadratic_ensemble_predictions.shape)

#The Predicted versus Actual Power Values are graphed for Ensemble LSTM with Quadratic Splining
plt.figure(2)
plt.title("Quadratic Splining Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("15 Minute Timesteps")
plt.ylabel("Power Value")
length = actual_values.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,actual_values,label = "Actual")
plt.plot(timesteps,quadratic_ensemble_predictions,label = "Predicted")
plt.text(.5, .5, 'Date Range = ' + str(starting_date_string) + " to " + str(ending_date_string), fontsize = 12)
plt.legend(loc = 'upper center')

#This gets input data from the second quarter of the year
#This input data is used to test the ensembleFuturePredictor Function
num_rows_linear = np.shape(data_weather_power)[0]
num_rows_quadratic = np.shape(quadratic_data)[0]
linear_indexes = np.arange(int(num_rows_linear * 0.25),int(num_rows_linear * 0.25 + 30))
quadratic_indexes = np.arange(int(num_rows_quadratic* 0.25),int(num_rows_quadratic * 0.25 + 30))

input_data_linear = data_weather_power[linear_indexes,:]
input_data_quadratic = quadratic_data[quadratic_indexes,:]

date = date_array[int(num_rows_linear * 0.25),:]
date_string = str(int(date[0])) + "/" + str(int(date[1])) + " hour = " + str(int(date[2]))+" "+"minute  = " +str(int(date[3]))

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
plt.xlabel("15 Minute Timesteps")
plt.ylabel("Power Values")
num_prev_points = np.shape(input_data_linear)[0]
plt.scatter(np.arange(0,num_prev_points) , input_data_linear[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_linear)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_linear,label = "Future Predictions" )
plt.legend(loc = "upper center")
plt.text(.5, .5, 'The Date = ' + str(date_string) , fontsize = 12)


#Graphs the output of the ensembleFuturePredictor function
#plots the future power predictions for LSTM with Quadratic Splining
plt.figure(4)
plt.title("Future Power Predictions from Ensemble LSTM with Quadratic Splining")
plt.xlabel("15 Minute Timesteps")
plt.ylabel("Power Values")
num_prev_points = np.shape(input_data_quadratic)[0]
plt.scatter(np.arange(0,num_prev_points) , input_data_quadratic[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_quadratic)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_quadratic,label = "Future Predictions" )
plt.legend(loc = "upper center")
plt.text(.5, .5, 'The Date = ' + str(date_string) , fontsize = 12)

plt.show()
print("Linear ensemble predictions = " , linear_ensemble_predictions)
print("Quadratic ensemble predictions = ", quadratic_ensemble_predictions)








