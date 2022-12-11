import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
from keras.models import load_model
import math
import matplotlib.pyplot as plt
from important_functions import partition_set,normalize_values,splineQuadratic

def ensemblePredictor(input_data,model,complete_dataset,num_timesteps):
    input = input_data
    num_rows = np.shape(input)[0]
    num_cols = np.shape(input)[1]
    predictions_array = np.array([])
    min_array = [0 for iter_var in range(5)]
    max_array = [np.max(complete_dataset[:,iter_var]) for iter_var in range(5)]

    for iter in np.arange(0,num_cols):
        input[:,iter] =  (input[:,iter] - min_array[iter])/(max_array[iter] - min_array[iter])
   
    input = input[:,np.arange(0,4)]
    predictor_input = []
    
    predictor_input.append(input[num_rows - 30: num_rows, :])
    predictor_input = np.array(predictor_input)
    print("predictor_input shape = ",np.shape(predictor_input))
    predicted_power = model.predict(predictor_input)

    predictions_array = np.append(predictions_array,predicted_power)
    
   
    power_input = np.transpose( [input_data[:,4]] )
    print("predicted power shape = ", np.shape(predicted_power))
    print("power inpiut shape = ",np.shape(power_input))
    power_input = np.append(power_input,predicted_power,axis = 0)

    for timestep in np.arange(0,num_timesteps - 1):
        current_input_size = np.shape(power_input)[0]

        predictor_input = []
        predictor_input.append(power_input[current_input_size - 30: current_input_size, :])
        predictor_input = np.array(predictor_input)
        predicted_power = power_model.predict(predictor_input)
        predictions_array = np.append(predictions_array,predicted_power)
        power_input = np.append(power_input,predicted_power,axis = 0)

    predictions_array = predictions_array * (max_array[4] - min_array[4]) + min_array[4]
    return predictions_array


    


data_weather_power = np.loadtxt("../power_weather_data/PowerVsWeather.csv",delimiter = ",",skiprows = 1,dtype = float)
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
quadratic_data = splineQuadratic(data_weather_power)

normalized_linear_data = normalize_values(data_weather_power)
normalized_quadratic_data = normalize_values(quadratic_data)


linear_max_array = [np.max(data_weather_power[:,iter_var]) for iter_var in range(5)]
linear_min_array = [0 for iter_var in range(5)]

quadratic_max_array = [np.max(quadratic_data[:,iter_var]) for iter_var in range(5)]
quadratic_min_array = [0 for iter_var in range(5)]

weather_predictions = np.loadtxt("linear_predictions.txt",delimiter= ",",dtype = float)
linear_ensemble_model = load_model("linear_LSTM.keras")
power_model = load_model("power_model.keras")

power_values = np.transpose ( [ normalized_linear_data[:,4] ] )
unused1,x_test,unused2,y_test = partition_set(weather_predictions,power_values)

ensemble_predictions = linear_ensemble_model.predict(x_test)
print("y_test shape = " , y_test.shape)
print("ensemble_predictions shape = " , ensemble_predictions.shape)


plt.figure(1)

plt.title("Linear Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("timestep")
plt.ylabel("power value")
length = y_test.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,y_test,label = "Actual")
plt.plot(timesteps,ensemble_predictions,label = "Predicted")
plt.legend(loc = 'upper center')




weather_predictions = np.loadtxt("quadratic_predictions.txt",delimiter= ",",dtype = float)
quadratic_ensemble_model = load_model("quadratic_LSTM.keras")
power_values = np.transpose ( [ normalized_quadratic_data[:,4] ] )
unused1,x_test,unused2,y_test = partition_set(weather_predictions,power_values)

ensemble_predictions = quadratic_ensemble_model.predict(x_test)
print("y_test shape = " , y_test.shape)
print("ensemble_predictions shape = " , ensemble_predictions.shape)

plt.figure(2)
plt.title("Quadratic Ensemble Model Predicted Vs. Actual Power Values")
plt.xlabel("timestep")
plt.ylabel("power value")
length = y_test.shape[0]
timesteps = np.transpose([np.arange(0,length)])
plt.plot(timesteps,y_test,label = "Actual")
plt.plot(timesteps,ensemble_predictions,label = "Predicted")
plt.legend(loc = 'upper center')
plt.show()

input_data_linear = np.ones((30,1))
for i in np.arange(0,5):
    random_column_data = np.random.uniform(low = linear_min_array[i],high = linear_max_array[i],size = (30,1))
    input_data_linear = np.append(input_data_linear,random_column_data,axis = 1)

input_data_quadratic = np.ones((30,1))
for i in np.arange(0,5):
    random_column_data = np.random.uniform(low = quadratic_min_array[i],high = quadratic_max_array[i],size = (30,1))
    input_data_quadratic = np.append(input_data_quadratic,random_column_data,axis = 1)
    
input_data_linear = np.delete(input_data_linear,0, axis = 1)
input_data_quadratic = np.delete(input_data_quadratic,0, axis = 1)
    


power_prediction_linear = ensemblePredictor(input_data_linear,linear_ensemble_model,data_weather_power,5)
power_prediction_quadratic = ensemblePredictor(input_data_quadratic,quadratic_ensemble_model,quadratic_data,5)
# print(linear_max_array)
# print(linear_min_array)
# print(quadratic_max_array)
# print(quadratic_min_array)

# print(power_prediction_linear)
plt.figure(3)
plt.title("Future Power Predictions from Linear Ensemble LSTM")
plt.xlabel("timestep")
plt.ylabel("power values")
num_prev_points = np.shape(input_data_linear)[0]

plt.scatter(np.arange(0,num_prev_points) , input_data_linear[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_linear)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_linear,label = "Future Predictions" )

plt.show()

plt.figure(4)
plt.title("Future Power Predictions from Quadratic Ensemble LSTM")
plt.xlabel("timestep")
plt.ylabel("power values")
num_prev_points = np.shape(input_data_quadratic)[0]

plt.scatter(np.arange(0,num_prev_points) , input_data_quadratic[:,4] , label = "Previous Data" )
num_future_points = len(power_prediction_quadratic)
plt.scatter(np.arange(num_prev_points, num_prev_points + num_future_points), power_prediction_quadratic,label = "Future Predictions" )

plt.show()









