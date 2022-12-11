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
# np.savetxt("weather_power_test.txt",data_weather_power)
# np.savetxt("quad_data.txt",quad_data)

data_weather_power = normalize_values(data_weather_power)
quadratic_data = splineQuadratic(data_weather_power)
print(quadratic_data)


# print(data_weather_power)
# np.savetxt("test_format.csv",data_weather_power,delimiter=",",fmt= "%1.4f %1.4f %1.4f %1.4f %1.4f")
for iter in range(2):
    transformed_data = []
    title = ""
    title1 = ""
    if(iter == 0):
        transformed_data = data_weather_power
        title = "Validation Error For Linear LSTMS"
        title1 = "Linear Ensemble LSTM training error"

    if(iter == 1):
        transformed_data = quadratic_data
        title = "Validation Error For Quadratic LSTMS"
        title1 = "Quadratic Ensemble LSTM training error"

    insolation_values =   np.transpose( [ transformed_data[:,0] ] ) 
    ambient_temperature =  np.transpose( [ transformed_data[:,1] ] ) 
    module_temperature =   np.transpose( [ transformed_data[:,2] ] )
    wind_velocity = np.transpose( [ transformed_data[:,3] ] )
    power_values = np.transpose ( [ transformed_data[:,4] ] )
    

    insolation_predict,insolation_model,error_insolation= universal_LSTM(insolation_values,insolation_values,False,"Insolation LSTM ")
    ambient_predict,ambient_model,error_ambient = universal_LSTM(ambient_temperature,ambient_temperature,False,"Ambient Temperature LSTM")
    module_predict,module_model,error_module = universal_LSTM(module_temperature,module_temperature,False,"Module Temperature LSTM")
    wind_predict,wind_model,error_wind = universal_LSTM(wind_velocity,wind_velocity,False,"Wind Velocity LSTM")
    power_predict,power_model,error_power = universal_LSTM(power_values,power_values,False,"Power LSTM")
    
    power_model.save("power_model.keras")

    weather_predictions = np.append(insolation_predict,ambient_predict,axis = 1)
    weather_predictions = np.append(weather_predictions,module_predict,axis = 1)
    weather_predictions = np.append(weather_predictions,wind_predict,axis = 1 )

    

    feature_data = transformed_data[:,np.arange(0,4)]
    ensemble_predictions,ensemble_model,ensemble_error= universal_LSTM(feature_data,power_values,True,title1)
    if(iter == 0):
        np.savetxt("linear_predictions.txt",weather_predictions,delimiter = ",")
        ensemble_model.save("linear_LSTM.keras")
    if(iter == 1):
        np.savetxt("quadratic_predictions.txt",weather_predictions,delimiter = ",")
        ensemble_model.save("quadratic_LSTM.keras")

    error_array = np.array([error_insolation,error_ambient,error_module,error_wind,ensemble_error])
    plt.figure()
    plt.title(title)
    plt.ylabel("Validation Error")
    plt.bar(["insolation","ambient temperature","module_temperature","wind_velocity","ensemble_model"],error_array)
    
   
plt.show()

