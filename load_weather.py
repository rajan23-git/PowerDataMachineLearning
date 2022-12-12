import numpy as np
import os

#This function performs Linear Splining on the weather data
#Weather datapoints are every 1 hour
#I used Linear Splining to estimate the datapoints at 15 minute intervals
def trasformMatrix(matrix,current_month):
    print("matrix shape",np.shape(matrix))
    modMatrix = np.ones((1,np.shape(matrix)[1] -1 + 4),dtype = float) 
    
    for iter in range(np.shape(matrix)[0] - 1):
       #the current date information is obtained
       current_date_time = matrix[iter,0]
       current_month = current_month
       current_day = int( (current_date_time.split("/"))[1] )
       current_hour = iter %24
       current_minute = 0
       
      
       #the lower and upper bounds are used to define the slope of Linear Equation
       lower_bound = matrix[iter,np.arange(1,np.shape(matrix)[1])]
       lower_bound = lower_bound.astype(float)
       upper_bound = matrix[iter + 1,np.arange(1,np.shape(matrix)[1])]
       upper_bound = upper_bound.astype(float)
       #the rate of change is calculated as the slope between the lower and upper bound
       slope = upper_bound - lower_bound
       #these are the datapoints added in between 15 minute intervals
       #these datapoints are calculated using the linear slope
       row_one = np.append( lower_bound ,[current_month,current_day,current_hour,current_minute])
       row_two = np.append( lower_bound + slope *0.25 ,[current_month,current_day,current_hour,current_minute + 15])
       row_three = np.append( lower_bound + slope *0.50 ,[current_month,current_day,current_hour,current_minute + 30])
       row_four = np.append( lower_bound + slope *0.75 ,[current_month,current_day,current_hour,current_minute + 45])
       #these datapoints are inserted in between datapoints of original data
       modMatrix = np.append(modMatrix,[row_one],axis = 0)
       modMatrix = np.append(modMatrix,[row_two],axis = 0)
       modMatrix = np.append(modMatrix,[row_three],axis = 0)
       modMatrix = np.append(modMatrix,[row_four],axis = 0)
    modMatrix = np.delete(modMatrix,0,axis = 0)
    return modMatrix   
       

different_months = np.array(["01","02","03","04","05","06","07","08","09","10","11"])
weather_filepath_array  = np.array([])
insolation_filepath_array = np.array([])
weather_data = np.array([[0,0,0,0,0,0,0]],dtype = float)
insolation_data = np.array([[0,0,0,0,0]],dtype = float)

total_hours = 0
total_insolation_datapoints = 0
total_power_datapoints = 0
#Code goes through each month folder of the Conditions Folder and the Production Folder
#builds a file containing all the weather data in the entire PVSystem folder
for different_month in different_months:
    #the filpaths of each weather file are obtained
    month_folder_path = "../PVSystem"+"/"+str(different_month)+"/" + "Conditions"
    weather_files = os.listdir(month_folder_path)
    #the filepaths of each power output file are obtained
    power_files = os.listdir("../PVSystem"+"/"+str(different_month)+"/" + "Production")
    # print(weather_files,"  Month = ", different_month )
    #counting the amount of power datapoints
    for power_file in power_files:
        path = "../PVSystem"+"/"+str(different_month)+"/" + "Production" +"/"+ str(power_file)
        file_power = np.loadtxt(path,delimiter=";",dtype=str)
        total_power_datapoints = total_power_datapoints +np.shape(file_power)[0]

    for weather_file in weather_files:
        file_path = month_folder_path + "/" + str(weather_file)
        #if statement selects weather files containing ambient temperature, module temperature, and wind_velocity
        if("Weather_Diagram_1" in weather_file):
            weather_filepath_array = np.append(weather_filepath_array,file_path)
            file_weather = np.loadtxt(file_path,delimiter = ";",dtype = str,skiprows=1)
            #calculated total number of weather datapoints or hours
            total_hours = total_hours + np.shape(file_weather)[0]
            #all the blank data is replaced with 0.00
            file_weather[file_weather == ''] = "0.00"
            file_weather[:,1] = (file_weather[:,1]).astype(float)
            file_weather[:,2] = (file_weather[:,2]).astype(float)
            file_weather[:,3] = (file_weather[:,3]).astype(float)
            current_month = int(different_month)
            #Linear Splining performed on weather data file
            file_weather = trasformMatrix(file_weather,current_month)
            #file data is appended to data matrix containing all weather data from all files
            weather_data = np.append(weather_data,file_weather,axis = 0)
        #If statement selects weather files containing only insolation data
        elif("Weather_Diagram_2" in weather_file):
            insolation_filepath_array = np.append(insolation_filepath_array,file_path)
            file_insolation = np.loadtxt(file_path,delimiter = ";",dtype = str,skiprows=1)
            #total number of insolation datapoints is calculated
            total_insolation_datapoints = total_insolation_datapoints + np.shape(file_insolation)[0]
            file_insolation[file_insolation == ''] = "0.00"
            file_insolation[:,1] = (file_insolation[:,1]).astype(float)
            current_month = int(different_month)
            #Linear Splining on insolation data
            file_insolation = trasformMatrix(file_insolation,current_month)
            #insolation file appended to insolation data matrix, containing all insolation data from all files
            insolation_data = np.append(insolation_data,file_insolation,axis = 0)

weather_data = np.delete(weather_data,0,axis = 0)
insolation_data = np.delete(insolation_data,0,axis = 0)


print("weather filepath array", weather_filepath_array)
print("insolation filepath array", insolation_filepath_array)
#weather data is sorted based on month, day, and hour
ind = np.lexsort((weather_data[:,5],weather_data[:,4],weather_data[:,3]))
weather_data = weather_data[ind]
#insolation data is sorted by month,day,and hour
ind = np.lexsort((insolation_data[:,3],insolation_data[:,2],insolation_data[:,1]))
insolation_data = insolation_data[ind]
#weather and insolation data are joined together columnwise, matching their dates together
weather_insolation_data = np.append(np.transpose([insolation_data[:,0]]),weather_data,axis = 1)

np.savetxt("weather_insolation_file.csv",weather_insolation_data,delimiter = ",",header = "Insolation,Ambient Temperature,Module temperature,WindVel,Month,Day,Hour,Minute",fmt = "%1.4f,%1.4f,%1.4f,%1.4f,%d,%d,%d,%d")


print("total hours",total_hours)
print("total power datapoints",total_power_datapoints)
print("total insolation points",total_insolation_datapoints)
