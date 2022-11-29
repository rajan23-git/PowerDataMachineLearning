import numpy as np
import os
def trasformMatrix(matrix,current_month):
    print("matrix shape",np.shape(matrix))
    modMatrix = np.ones((1,np.shape(matrix)[1] -1 + 4),dtype = float)
    
    for iter in range(np.shape(matrix)[0] - 1):
       current_date_time = matrix[iter,0]
       current_month = current_month
       current_day = int( (current_date_time.split("/"))[1] )
       current_hour = iter %24
       current_minute = 0
       
      
      
       lower_bound = matrix[iter,np.arange(1,np.shape(matrix)[1])]
       lower_bound = lower_bound.astype(float)
       upper_bound = matrix[iter + 1,np.arange(1,np.shape(matrix)[1])]
       upper_bound = upper_bound.astype(float)
       slope = upper_bound - lower_bound

       row_one = np.append( lower_bound ,[current_month,current_day,current_hour,current_minute])
       row_two = np.append( lower_bound + slope *0.25 ,[current_month,current_day,current_hour,current_minute + 15])
       row_three = np.append( lower_bound + slope *0.50 ,[current_month,current_day,current_hour,current_minute + 30])
       row_four = np.append( lower_bound + slope *0.75 ,[current_month,current_day,current_hour,current_minute + 45])
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
for different_month in different_months:
    month_folder_path = "../PVSystem"+"/"+str(different_month)+"/" + "Conditions"
    weather_files = os.listdir(month_folder_path)
    power_files = os.listdir("../PVSystem"+"/"+str(different_month)+"/" + "Production")
    # print(weather_files,"  Month = ", different_month )
    for power_file in power_files:
        path = "../PVSystem"+"/"+str(different_month)+"/" + "Production" +"/"+ str(power_file)
        file_power = np.loadtxt(path,delimiter=";",dtype=str)
        total_power_datapoints = total_power_datapoints +np.shape(file_power)[0]

    for weather_file in weather_files:
        file_path = month_folder_path + "/" + str(weather_file)
        if("Weather_Diagram_1" in weather_file):
            weather_filepath_array = np.append(weather_filepath_array,file_path)
            file_weather = np.loadtxt(file_path,delimiter = ";",dtype = str,skiprows=1)
            total_hours = total_hours + np.shape(file_weather)[0]
            file_weather[file_weather == ''] = "0.00"
            file_weather[:,1] = (file_weather[:,1]).astype(float)
            file_weather[:,2] = (file_weather[:,2]).astype(float)
            file_weather[:,3] = (file_weather[:,3]).astype(float)
            current_month = int(different_month)
            file_weather = trasformMatrix(file_weather,current_month)
            weather_data = np.append(weather_data,file_weather,axis = 0)
        elif("Weather_Diagram_2" in weather_file):
            insolation_filepath_array = np.append(insolation_filepath_array,file_path)
            file_insolation = np.loadtxt(file_path,delimiter = ";",dtype = str,skiprows=1)
            total_insolation_datapoints = total_insolation_datapoints + np.shape(file_insolation)[0]
            file_insolation[file_insolation == ''] = "0.00"
            file_insolation[:,1] = (file_insolation[:,1]).astype(float)
            current_month = int(different_month)
            file_insolation = trasformMatrix(file_insolation,current_month)
            insolation_data = np.append(insolation_data,file_insolation,axis = 0)

weather_data = np.delete(weather_data,0,axis = 0)
insolation_data = np.delete(insolation_data,0,axis = 0)


print("weather filepath array", weather_filepath_array)
print("insolation filepath array", insolation_filepath_array)
ind = np.lexsort((weather_data[:,5],weather_data[:,4],weather_data[:,3]))
weather_data = weather_data[ind]

ind = np.lexsort((insolation_data[:,3],insolation_data[:,2],insolation_data[:,1]))
insolation_data = insolation_data[ind]

weather_insolation_data = np.append(np.transpose([insolation_data[:,0]]),weather_data,axis = 1)
# column_indexes = np.array([0,1,2,3])
# for column_index in column_indexes:

#     for iter in range(np.shape(weather_insolation_data)[0]):
#         current_insolation = weather_insolation_data[iter][column_index]
#         if(current_insolation == 0):
#             non_zeros_indexes = (np.where (weather_insolation_data[:,column_index] !=0))[0]
#             left_indexes = non_zeros_indexes[np.where((non_zeros_indexes - iter) < 0 )]
#             right_indexes = non_zeros_indexes[np.where((non_zeros_indexes - iter) > 0)]
#             print("iter",iter)
#             print("left_indexes",left_indexes)
#             print("right_indexes",right_indexes)
#             index_left = -100000
#             index_right = -10000
#             if(len(left_indexes) == 0):
#                 index_left = 0
#             else:
#                 index_left = np.max(left_indexes)
#             if(len(right_indexes) == 0):
#                 index_right = np.shape(weather_insolation_data)[0] - 1
#             else:
#                 index_right = np.min(right_indexes)


#             slope = (weather_insolation_data[index_right][column_index] - weather_insolation_data[index_left][column_index])/(index_right - index_left)
#             adjusted_insolation = weather_insolation_data[index_left][column_index] + slope * (iter - index_left)
#             weather_insolation_data[iter][column_index] = adjusted_insolation


#             print(non_zeros_indexes)


np.savetxt("weather_insolation_file.csv",weather_insolation_data,delimiter = ",",header = "Insolation,Ambient Temperature,Module temperature,WindVel,Month,Day,Hour,Minute",fmt = "%1.4f,%1.4f,%1.4f,%1.4f,%d,%d,%d,%d")



        
        


print("total hours",total_hours)
print("total power datapoints",total_power_datapoints)
print("total insolation points",total_insolation_datapoints)
