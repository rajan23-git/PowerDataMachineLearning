import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
import math
import matplotlib.pyplot as plt
from important_functions import normalize_values,partition_set,splineQuadratic,universal_LSTM


data_weather_power = np.loadtxt("../power_weather_data/PowerVsWeather.csv",delimiter = ",",skiprows = 1,dtype = float)
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
print("data weather power  = ", data_weather_power)

#data is normalized
data_weather_power = normalize_values(data_weather_power)
#Quadratic Splining is performed on data
quadratic_data = splineQuadratic(data_weather_power)
print(quadratic_data)

for iter in range(2):
    transformed_data = []
    title = ""
    title1 = ""
    #LSTMs are run on both the linear splined data, and the quadratic splined data
    if(iter == 0):
        #both linear and quadratic splined data are both processed
        #at different iterations of loop
        transformed_data = data_weather_power
        title = "Validation Error For LSTMs with Linear Splining"
        title1 = "Ensemble LSTM training error with Linear Splining"

    if(iter == 1):
        transformed_data = quadratic_data
        title = "Validation Error For LSTMs with Quadratic Splining"
        title1 = "Ensemble LSTM training error with Quadratic Splining"
    #data for different weather features
    insolation_values =   np.transpose( [ transformed_data[:,0] ] ) 
    ambient_temperature =  np.transpose( [ transformed_data[:,1] ] ) 
    module_temperature =   np.transpose( [ transformed_data[:,2] ] )
    wind_velocity = np.transpose( [ transformed_data[:,3] ] )
    power_values = np.transpose ( [ transformed_data[:,4] ] )
    
    #individual LSTMs are built for each weather feature
    insolation_predict,insolation_model,error_insolation= universal_LSTM(insolation_values,insolation_values,False,"Insolation LSTM ")
    ambient_predict,ambient_model,error_ambient = universal_LSTM(ambient_temperature,ambient_temperature,False,"Ambient Temperature LSTM")
    module_predict,module_model,error_module = universal_LSTM(module_temperature,module_temperature,False,"Module Temperature LSTM")
    wind_predict,wind_model,error_wind = universal_LSTM(wind_velocity,wind_velocity,False,"Wind Velocity LSTM")
    power_predict,power_model,error_power = universal_LSTM(power_values,power_values,False,"Power LSTM")
    #the LSTM models for each weather feature are saved
    insolation_model.save("insolation_model.keras")
    ambient_model.save("ambient_model.keras")
    module_model.save("module_model.keras")
    wind_model.save("wind_model.keras")
    power_model.save("power_model.keras")
    #predictions from each individual LSTM are combined together
    weather_predictions = np.append(insolation_predict,ambient_predict,axis = 1)
    weather_predictions = np.append(weather_predictions,module_predict,axis = 1)
    weather_predictions = np.append(weather_predictions,wind_predict,axis = 1 )

    unused1,unused2,unused3,actual_power_values = partition_set(power_values,power_values)
    #these are the actual power values that will be used for validation
    weather_predictions = np.append(weather_predictions,actual_power_values,axis = 1 )

    #Ensemble LSTM is built, whose input is weather feature data,
    #and whose output is Power 
    feature_data = transformed_data[:,np.arange(0,4)]
    #Ensemble LSTM trained on weather feature data
    ensemble_predictions,ensemble_model,ensemble_error= universal_LSTM(feature_data,power_values,True,title1)
    if(iter == 0):
        #prediction input for Ensemble LSTM with Linear Splining is saved
        np.savetxt("linear_predictions.txt",weather_predictions,delimiter = ",")
        #Ensemble LSTM model with Linear Splining is saved
        ensemble_model.save("linear_LSTM.keras")
    if(iter == 1):
        #prediction input for Ensemble LSTM with Quadratic Splining is saved
        np.savetxt("quadratic_predictions.txt",weather_predictions,delimiter = ",")
        #Ensemble LSTM model with Quadratic Splining is saved
        ensemble_model.save("quadratic_LSTM.keras")
    #The validation error from each individual LSTM is plotted in a bar graph
    #validation error of Ensemble model also plotted
    error_array = np.array([error_insolation,error_ambient,error_module,error_wind,ensemble_error])
    plt.figure()
    plt.title(title)
    plt.xlabel("LSTM type")
    plt.ylabel("Validation Error")
    plt.bar(["insolation","ambient ","module","wind_velocity","Ensemble"],error_array)
       
   
plt.show()

