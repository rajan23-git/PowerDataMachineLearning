import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
import math
import matplotlib.pyplot as plt
class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        



def splineQuadratic(data_matrix):
    orig_rows = np.shape(data_matrix)[0]
    transformed_matrix = np.ones((31896,1),dtype = float)

    data_matrix = data_matrix[np.arange(0,orig_rows,4),:]
    print("arange quadratic  = ",len(np.arange(0,orig_rows,4)))
    new_rows = np.shape(data_matrix)[0]
    new_cols = np.shape(data_matrix)[1]
    for column_iterator in np.arange(0,new_cols):
        new_column = []
        #3988
        for row_iterator in np.arange(0,new_rows -2,2):
            point_one = Point(x = row_iterator, y = data_matrix[row_iterator][column_iterator])
            point_two = Point(x = row_iterator + 1 ,y =  data_matrix[row_iterator + 1][column_iterator])
            point_three = Point(x = row_iterator +2 , y = data_matrix[row_iterator + 2][column_iterator])
                        
            coefficients = np.array([[1,1,1]],dtype = float)
            coefficients = np.append(coefficients, [[(point_one.x)**2, point_one.x,1]],axis = 0)
            coefficients = np.append(coefficients, [[(point_two.x)**2, point_two.x,1]], axis = 0)
            coefficients = np.append(coefficients, [[(point_three.x)**2, point_three.x,1]], axis = 0)
            coefficients = np.delete(coefficients,0,axis = 0)

            y_values = np.array([point_one.y,point_two.y,point_three.y])
            quadratic_coefficients = np.linalg.solve(coefficients,y_values)
            A = quadratic_coefficients[0]
            B = quadratic_coefficients[1]
            C = quadratic_coefficients[2]
            in_between_points = np.array([])
            for incrementor in np.array([0.25,0.5,0.75,1.25,1.5,1.75]):
                x_value = row_iterator + incrementor
                in_between_point = A * (x_value)**2 + B*(x_value) + C
                in_between_points = np.append(in_between_points,in_between_point)

            new_column = np.append(new_column,[point_one.y])
            new_column = np.append(new_column,in_between_points[0:3])
            new_column = np.append(new_column,[point_two.y])
            new_column = np.append(new_column,in_between_points[3:6])


        new_column = np.transpose([new_column])
        transformed_matrix = np.append(transformed_matrix,new_column,axis = 1)
    # np.savetxt("test_matrix",transformed_matrix)
    transformed_matrix = np.delete(transformed_matrix,0,axis = 1)
    return transformed_matrix







def normalize_values(data_matrix):
    for iter in np.arange(0,np.shape(data_matrix)[1]):    
        data_matrix[:,iter] = (data_matrix[:,iter] - np.max(data_matrix[:,iter]))/(np.max(data_matrix[:,iter]) - np.min(data_matrix[:,iter]))
    return data_matrix



def universal_LSTM(x_data,y_data,fig_number,title):
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
    var_model = model.fit(x_train, y_train, batch_size= 2000, epochs=num_epochs )
    var_predictions = model.predict(x_test)
    
    print("single var prediction = " ,var_predictions)
    print("single var predictions shape = ",np.shape(var_predictions ))
    print("actual shape = ",np.shape(y_test ))

    plt.figure(fig_number)
    plt.subplot(121)
    plt.title(title)
    plt.xlabel("Epoch Number")
    plt.ylabel("Error Over Epochs")
    plt.scatter(np.arange(0,num_epochs),var_model.history['loss'])
    plt.plot(np.arange(0,num_epochs),var_model.history['loss'])
    plt.subplot(122)
    plt.title("Predicted " + str(title) + " values over time")
    plt.xlabel("Time")
    plt.ylabel(str(title) + "Data")
    plt.scatter(np.arange(np.shape(var_predictions)[0]),var_predictions.flatten(), label = "predicted")
    plt.scatter(np.arange(np.shape(y_test)[0]),y_test.flatten(), label = "actual")
    plt.legend(loc = 'upper right')
    plt.show()

    return var_predictions,model


data_weather_power = np.loadtxt("../power_weather_data/PowerVsWeather.csv",delimiter = ",",skiprows = 1,dtype = float)
data_weather_power = np.delete(data_weather_power,[0,5,6,7,8],axis = 1)
print("data weather power  = ", data_weather_power)
# np.savetxt("weather_power_test.txt",data_weather_power)
quadratic_data = splineQuadratic(data_weather_power)
print(quadratic_data)
# np.savetxt("quad_data.txt",quad_data)

data_weather_power = normalize_values(data_weather_power)
# print(data_weather_power)
# np.savetxt("test_format.csv",data_weather_power,delimiter=",",fmt= "%1.4f %1.4f %1.4f %1.4f %1.4f")
insolation_values =   np.transpose( [ data_weather_power[:,0] ] ) 
ambient_temperature =  np.transpose( [ data_weather_power[:,1] ] ) 
module_temperature =   np.transpose( [ data_weather_power[:,2] ] )
wind_velocity = np.transpose( [ data_weather_power[:,3] ] )
power_values = np.transpose ( [ data_weather_power[:,4] ] )


insolation_predict,insolation_model = universal_LSTM(insolation_values,insolation_values,1,"Insolation LSTM ")
ambient_predict,ambient_model = universal_LSTM(ambient_temperature,ambient_temperature,2,"Ambient Temperature LSTM")
module_predict,module_model = universal_LSTM(module_temperature,module_temperature,3,"Module Temperature LSTM")
wind_predict,wind_model = universal_LSTM(wind_velocity,wind_velocity,4,"Wind Velocity LSTM")
combined_predictions = np.append(insolation_predict,ambient_predict,axis = 1)
combined_predictions = np.append(combined_predictions,module_predict,axis = 1)
combined_predictions = np.append(combined_predictions,wind_predict,axis = 1 )
print("combined predictions shape = ", np.shape(combined_predictions))
ensemble_predictions,ensemble_model = universal_LSTM(combined_predictions,power_values,5,"Ensemble LSTM for Power")




