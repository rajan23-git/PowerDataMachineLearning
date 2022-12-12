import numpy as np
import math
import sys
import numpy as np
from tensorflow import keras
import keras.layers as layers
import matplotlib.pyplot as plt
#this class contains the X and Y coordinates of a point
class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
#this function performs Quadratic Splining on Weather Data
#weather datapoints are recorded every 1 hour
#uses Quadratic fitting to estimate the datapoints at 15 minute intervals
def splineQuadratic(data_matrix):
    orig_rows = np.shape(data_matrix)[0]
    transformed_matrix = np.ones((31896,1),dtype = float)

    data_matrix = data_matrix[np.arange(0,orig_rows,4),:]
    print("arange quadratic  = ",len(np.arange(0,orig_rows,4)))
    new_rows = np.shape(data_matrix)[0]
    new_cols = np.shape(data_matrix)[1]
    for column_iterator in np.arange(0,new_cols):
        new_column = []

        for row_iterator in np.arange(0,new_rows -2,2):
            #obtains three Points, to use for Quadratic fitting
            point_one = Point(x = row_iterator, y = data_matrix[row_iterator][column_iterator])
            point_two = Point(x = row_iterator + 1 ,y =  data_matrix[row_iterator + 1][column_iterator])
            point_three = Point(x = row_iterator +2 , y = data_matrix[row_iterator + 2][column_iterator])
            #coefficients array contains the value of x^2 and x 
            #this coefficients array contains coefficients in a system of equations            
            coefficients = np.array([[1,1,1]],dtype = float)
            coefficients = np.append(coefficients, [[(point_one.x)**2, point_one.x,1]],axis = 0)
            coefficients = np.append(coefficients, [[(point_two.x)**2, point_two.x,1]], axis = 0)
            coefficients = np.append(coefficients, [[(point_three.x)**2, point_three.x,1]], axis = 0)
            coefficients = np.delete(coefficients,0,axis = 0)
            #y values contain the y_values of the points
            y_values = np.array([point_one.y,point_two.y,point_three.y])
            #The A,B,C coefficients of a Quadratic Equation are calculated using Linear Algebra and System of Equations
            quadratic_coefficients = np.linalg.solve(coefficients,y_values)
            A = quadratic_coefficients[0]
            B = quadratic_coefficients[1]
            C = quadratic_coefficients[2]
            in_between_points = np.array([])
            #the datapoints in between the three original points are estimated, using the quadratic equation 
            for incrementor in np.array([0.25,0.5,0.75,1.25,1.5,1.75]):
                x_value = row_iterator + incrementor
                in_between_point = A * (x_value)**2 + B*(x_value) + C
                in_between_points = np.append(in_between_points,in_between_point)
            #the in_between datapoints are inserted into the original data
            new_column = np.append(new_column,[point_one.y])
            new_column = np.append(new_column,in_between_points[0:3])
            new_column = np.append(new_column,[point_two.y])
            new_column = np.append(new_column,in_between_points[3:6])


        new_column = np.transpose([new_column])
        transformed_matrix = np.append(transformed_matrix,new_column,axis = 1)
    # np.savetxt("test_matrix",transformed_matrix)
    transformed_matrix = np.delete(transformed_matrix,0,axis = 1)
    return transformed_matrix


#this function normalizes the data, using min-max normalization
#normalizes each column, or each weather feature
def normalize_values(data_matrix):
    transformed_matrix = np.copy(data_matrix)
    for iter in np.arange(0,np.shape(data_matrix)[1]):    
        transformed_matrix[:,iter] = (transformed_matrix[:,iter] - np.min(transformed_matrix[:,iter]))/(np.max(transformed_matrix[:,iter]) - np.min(transformed_matrix[:,iter]))
    return transformed_matrix

#this function splits up the data into x_train,y_test,y_train,y_test    
def partition_set(x_data,y_data):
    num_rows = np.shape(x_data)[0]
    #training and validation are split up 80 and 20 percent
    training_split = math.floor( num_rows * 0.80 )
    x_train = []
    y_train =   y_data[np.arange(30,training_split),:]
    
    x_test = []
    y_test =  y_data[np.arange(training_split - 30,num_rows),:]
    #x_train data consists of groups of size 30
    #the x_train data consists of the past 30 datapoints
    #these previous 30 datapoints are used to predict a single future datapoint
    for iterator in np.arange(30,training_split):
        appended_row = x_data[(iterator-30):iterator, :]
        x_train.append(appended_row)
    #x_test data has groups of 30 datapoints
    #each group of size 30 is used to predict a future datapoint
    for iter in np.arange(training_split - 30,num_rows):
        appended_row = x_data[iter-30 : iter, :]
        x_test.append(appended_row)
    #x_train,x_test,y_train,y_test are converted to numpy arrays for the LSTM 
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1],np.shape(x_train)[2]))
    x_test = np.reshape(x_test, (np.shape(x_test)[0],np.shape(x_test)[1],np.shape(x_test)[2]))
    return x_train,x_test,y_train,y_test



#this function trains an LSTM model given input x_data and output y_data
def universal_LSTM(x_data,y_data,displayGraph,title):
    x_train,x_test,y_train,y_test = partition_set(x_data,y_data)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #model with 2 LSTM layers, one dense layer, and an output dense layer
    #is constructed
    model = keras.Sequential()
    model.add(layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2] )))
    model.add(layers.LSTM(50, return_sequences=False))
    model.add(layers.Dense(30))
    model.add(layers.Dense(np.shape(y_data)[1]))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics = ['accuracy'])
    num_epochs = 5
    #model is trained 
    var_model = model.fit(x_train, y_train, batch_size= 1500, epochs=num_epochs )
    #model is used to predict output data
    var_predictions = model.predict(x_test)
    
    print("single var prediction = " ,var_predictions)
    print("single var predictions shape = ",np.shape(var_predictions ))
    print("actual shape = ",np.shape(y_test ))
    #if displayGraph boolean is True, training error over time is graphed
    if(displayGraph == True):
        plt.figure()
        plt.title(title)
        plt.xlabel("Epoch Number")
        plt.ylabel("Error Over Epochs")
        plt.scatter(np.arange(0,num_epochs),var_model.history['loss'])
        plt.plot(np.arange(0,num_epochs),var_model.history['loss'])

    #validation error is calulated based on y_test and predicted values
    validation_error = math.sqrt(np.sum(np.power((var_predictions - y_test),2)) / len(var_predictions))
    return var_predictions,model,validation_error