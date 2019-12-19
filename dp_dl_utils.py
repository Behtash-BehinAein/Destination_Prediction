# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:39:54 2019

@author: Behtash Behin-Aein
"""
import os
import pickle
import pandas as pd
import numpy as np
import holidays
#import Geohash as gh
import pygeohash as gh
import category_encoders as ce
#import tensorflow as tf
from geopy import distance          # Used only to benchmark custom-defined function 
# -----------------------------------------------------------------------------

'''
Access the dataset and userfiles 
'''
def access_dataset(path):
    '''
    - Takes the path to the dataset holding trip data for all users
    - Returns 
        - os-invariant path to the dataset      | type: python pathlib path
        - a list containing the userfile names  | type: list 
    '''
    dataset = path       
    userFiles = [filename[:-5] for filename in os.listdir(dataset) if filename[-4:] == 'json']   # list of filenames for user data
    return dataset, userFiles

# =================================================================================================================================================
'''
Load data into a dictionary of {user-file-name : user-trip-dataframe}
'''
def dataset_to_user_dict(dataset, filenames):
    '''
    - Takes the path to the dataset and a list of user filenames 
    - Reads the json files into pandas dataframes 
    - Returns 
        - A dictionary with user filenames as the keys and user trips (dataframes) as the values | type : dict 
    
    - Additional notes 
        - lat,lon values have been cast as float32 for all other further processing
            -  5th&6th decimal place represent meters and fractions of meters
            -  The model built here is not intended to have accuracy of meters so the approximation is adequate    
    '''
    if type(filenames)!= type(list()) or len(filenames)==0:
        raise TypeError('Input should be a list with at least 1 filename in it')
    userTrips = {}   # Dict holds dataframes for each user 
    for user in filenames:
        userTrips[user] = pd.read_json(os.path.join(dataset, user+'.json'), lines = True).sort_values('start_time').reset_index(drop=True)
        # Some precision is lost; however, this is inline with the overall intended location accuracy of prediction 
        userTrips[user] = userTrips[user].astype({'start_location_lat':'float32', 'start_location_lon' :'float32', 'end_location_lat': 'float32', 'end_location_lon': 'float32'})         
        # Average out all lat,lon pairs indicating the same address name
        #    - This has some denoising effect  
        userTrips[user]['start_location_lat'] = userTrips[user]['start_location_lat'].groupby(userTrips[user]['start_location_name']).transform('mean')  
        userTrips[user]['start_location_lon'] = userTrips[user]['start_location_lon'].groupby(userTrips[user]['start_location_name']).transform('mean')  
        userTrips[user]['end_location_lat'] = userTrips[user]['end_location_lat'].groupby(userTrips[user]['end_location_name']).transform('mean')  
        userTrips[user]['end_location_lon'] = userTrips[user]['end_location_lon'].groupby(userTrips[user]['end_location_name']).transform('mean')  
    return userTrips
# =================================================================================================================================================
'''
Exract the last month of data for each user 
'''
def getLastMonth(filenames, userTrips):
    '''
    - Takes in userTrips dictionary of {user-file-name: user trip dataframe} 
    - Returns
        - Dict of last recorded month for each user | type: dict 
        - ASSUMES that the data for the last month is more or less complete (full month or close to it)
    '''
    lastMonth = {}
    for user in filenames:
        lastMonth[user] = int(userTrips[user]['end_time'].iloc[-1].date().strftime('%m'))
    return lastMonth    
# =================================================================================================================================================
'''
Asser that start and end time zones are the same 
'''
def checkTimeZone(filenames, userTrips):
    '''
    - Takes a list of user filenames holding and the userTrips dictionay of userTrip dataframes
    - Asserts that the start and end time zones of the trips are same for each user 
    - Resturns
        - no return, just asserts
    '''
    for user in filenames:
        if 'start_timezone' in userTrips[user].columns:
            assert sum((userTrips[user]['start_timezone'] == userTrips[user]['end_timezone'])) == len(userTrips[user])    
# =================================================================================================================================================
'''
Drop columns based on a user-defined drop list
'''   
def drop_cols(filenames, userTrips, dropList):
    '''
    - Takes a list of user filenames and the userTrips dictionary 
    - Drops the columns from userTrips based on user-defined dropList 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''  
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        df.drop(labels = dropList, inplace=True, axis=1, errors= 'ignore')
        Dict[user] = df 
    return Dict   
# =================================================================================================================================================
'''
# Converts lat,lon to geohashes 
'''
def latlon_to_geohash_single(lat, lon, precision=6):
    '''
    - Converts lat,lon to geohashes  where both lat and lot are iterables
    - Precision: designates geohash granularity
                - Precision 5: +/- 2.4 Km
                - Precision 6: +/- 0.61 Km
                - Precision 7: +/- 0.076 Km
                - Precision 8: +/- 0.019 Km  
    - Returns 
                - geohashes | type: string 
    '''
    #prc = {5: '2.4 Km' , 6: '0.61 Km' ,7 : '0.076 Km', 8 : '0.019 Km'}
    #print(f'Maximum error is {prc[precision]}')
    #return [gh.encode(i,j, precision = precision) for i,j in zip(lat,lon)]
    return gh.encode(lat,lon, precision = precision)
latlon_to_geohash = np.vectorize(latlon_to_geohash_single)
# =================================================================================================================================================
'''
# Add geohashes to the dataframes using latlon_to_geohash func 
'''
def addGeohash(filenames, userTrips, precision=6):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Adds geohashes to a deep-copy of userTrips dictionary
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        df['start_location_geohash'] = latlon_to_geohash(df['start_location_lat'], df['start_location_lon'], precision=precision )
        df['end_location_geohash']   = latlon_to_geohash(df['end_location_lat'], df['end_location_lon'], precision=precision )
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
Unify lat,lon values based on common geohashes 
'''           
  
def unifyLatLon(filenames, userTrips):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Unifies all lat,lon values based on shared geohashes 
        - Replaces all with the average of all
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable    
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        # All start lat,lon within the same geohash will be set to their average
        df['start_location_lat'] = df['start_location_lat'].groupby(df['start_location_geohash']).transform('mean')
        df['start_location_lon'] = df['start_location_lon'].groupby(df['start_location_geohash']).transform('mean')

        # All end lat,lon within the same geohash will be set to their average
        df['end_location_lat']   = df['end_location_lat'].groupby(df['end_location_geohash']).transform('mean')
        df['end_location_lon']   = df['end_location_lon'].groupby(df['end_location_geohash']).transform('mean')

        # end location lat,lon with the same geohash as the start location will be set start locatin lat,lon 
        df.loc[df['end_location_geohash'] == df['start_location_geohash'] , ['end_location_lat'] ] = df['start_location_lat']
        df.loc[df['end_location_geohash'] == df['start_location_geohash'] , ['end_location_lon'] ] = df['start_location_lon']
        
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
Calculate time delta 
'''           
def timeDelta(t1, t2):
    '''
    - Calculates the time difference in minutes between two Pandas datetime series
    - t1: Pandas Series of timestamps earlier than t2
    - t2: Pandas Series of timestamps of timestamps later than t1
    - Returns 
        - Series of time difference in whole minutes | type: Pandas Series of int's  
    '''
    delta = t2 - t1
    correct = pd._libs.tslibs.timestamps.Timestamp
    if type(t1[0])!=correct or type(t2[0])!= correct:
        raise TypeError('Entires of t1 and t2 must be Pandas timestamps')    
    if any(delta.dt.days<0):
        raise ValueError('All time instances in t2 must be later than or equal to time instances in t1')
    return (delta.dt.total_seconds()//60).astype('uint16')
# =================================================================================================================================================
'''
Add trip duration in min to each user's dataframe  
'''    
def addTripDur(filenames, userTrips):
    '''
    - Takes a list of user filenames and the userTrips dictionary 
    - Calculates trip duration
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''  
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        df['tripDur_min'] = timeDelta(df['start_time'], df['end_time'])
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
Calculate geodesic distance between two gelocations based on lat,lon
'''
def distanceGeodesic(lat1, lon1, lat2, lon2):
    '''
    - Calculates geodedic distance between two locations 
    - lat1, lon1, lat2, lon2 are iterables of geopoints  
    - Returns 
        - List of distances in kilometers | type: list of floats
    '''
    return round(distance.distance((lat1, lon1), (lat2, lon2)).kilometers, 3)
    #return [round(distance.distance((la1,lo1), (la2,lo2)).kilometers,3) for la1,lo1,la2,lo2 in zip(lat1, lon1, lat2, lon2)]   
vectorized_distance_geodesic = np.vectorize(distanceGeodesic)
# =================================================================================================================================================
'''
Calculate great-circle distance between two gelocations based on lat,lon
'''
def distanceGreatCircle(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two geolocations based on lat,lon
    """
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    d_lat = lat2 - lat1 
    d_lon = lon2 - lon1 

    # haversine formula 
    a = np.sin(d_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.round((c * 6371).astype('float'),3) # Radius of earth in Km: 6371
# =================================================================================================================================================
'''
Add trip distance in Km to each user's dataframe 
'''
def addTripDis(filenames, userTrips):
    '''
    - Takes a list of user filenames and the userTrips dictionary 
    - Calculates trip distances and adds them to the user dataframes 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''  
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        df['tripDis_Km'] = distanceGreatCircle(df['start_location_lat'], df['start_location_lon'], df['end_location_lat'], userTrips[user]['end_location_lon']).astype('float16')
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
# Save the processed data set at user-defined stages
'''
def save_proc_dataset(dataset, name, stage=None):
    '''
    - Takes the dataset and saves it to dataset_stage1.pkl
    - Use the stage parameter to save dataset at various stages 
    '''
    with open(f'{name}_stage{stage}.pkl', 'wb') as handle:     
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
# =================================================================================================================================================
'''
# Load the processed data set at user-defined stages
'''
def load_proc_dataset(name, stage=None): 
    '''
    - Takes the stage number and load the userTrips dictionary 
    - Use the stage parameter to load dataset at various stages 
    '''    
    with open(f'{name}_stage{stage}.pkl', 'rb') as handle:    
        dataset = pickle.load(handle)
    return dataset
# =================================================================================================================================================
'''
# Add last trip distance and duration as features 
'''
def addLastTripDisDir(filenames, userTrips, dis=True, dur=True):
    """
    - Takes a list of user filename and userTrips dictionary
    - Adds last trip duration and travel-distance as features to the current trip 
    - Input parameters "dis" and "dur" are both booleans with True as default.
        - If False, that feature is not added to the dataframe 
    - Returns
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    """
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        df['lastTripDis_Km'] = pd.Series([0]).append(df['tripDis_Km'][:-1].reset_index(drop=True)).reset_index(drop=True).astype('float16')
        df['lastTripDur_min'] = pd.Series([0]).append(df['tripDur_min'][:-1].reset_index(drop=True)).reset_index(drop=True).astype('uint16')
        Dict[user] = df
    return Dict 
# =================================================================================================================================================    
'''
# Add distance to last trip location as a feature
'''
def addEndToStartDis(filenames, userTrips):
    """
    - Takes a list of user filename and userTrips dictionary
    - Adds distance to the last trip end-location as a feature in Km 
    - Returns
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    """
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        endLat   = pd.Series([0]).append(df['end_location_lat'][:-1].reset_index(drop=True)).reset_index(drop=True)
        endLon   = pd.Series([0]).append(df['end_location_lon'][:-1].reset_index(drop=True)).reset_index(drop=True)
        startLat = pd.Series([0]).append(df['start_location_lat'][1:].reset_index(drop=True)).reset_index(drop=True)
        startLon = pd.Series([0]).append(df['start_location_lon'][1:].reset_index(drop=True)).reset_index(drop=True)

        df['endToStartDis_Km'] = distanceGreatCircle(endLat, endLon, startLat, startLon).astype('float16')
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
# Filter out trips with distances less than user defined value in Km
'''
def denoiseDis(filenames, userTrips, cols = ['lastTripDis[Km]', 'endToStartDis[Km]'], minDistance=0.1):
    '''
    - Takes a list of user filenames and the userTrips dictionary 
    - Sets distances in user-specified columns to 0 depending on the threshold value defined by the user
    - minDistance: user-defined threshold in Km.
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        for col in cols:
            df[col] = df[col].where(df[col]>0.1, other=0, try_cast=True)
        Dict[user] = df
    return Dict    
# =================================================================================================================================================
'''
# Add time passed since last trip ended as a feature
'''
def addEndToStartDur(filenames, userTrips):
    """
    - Takes a list of user filename and userTrips dictionary
    - Adds the time past since last trip ended as a feature in minutes  
    - Returns
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    """
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        lastEnd  = pd.Series(df['start_time'][0]).append(df['end_time'][:-1].reset_index(drop=True)).reset_index(drop=True)
        curStart = pd.Series(df['start_time'][0]).append(df['start_time'][1:].reset_index(drop=True)).reset_index(drop=True)
        df['endToStartDur_min'] = timeDelta(lastEnd,  curStart)
        Dict[user] = df
    return Dict  
# =================================================================================================================================================
'''
# Add time passed since last trip ended as a feature
'''
def addSameDay(filenames, userTrips):
    """
    - Takes a list of user filename and userTrips dictionary
    - Adds a boolean to indicate if the current trip is taking place on the same day as last   
    - Returns
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    """
    Dict = {}
    for user in filenames: 
        df = userTrips[user].copy(deep=True)
        lastEnd  = pd.Series(df['start_time'][0]).append(df['end_time'][:-1].reset_index(drop=True)).reset_index(drop=True)
        curStart = pd.Series(df['start_time'][0]).append(df['start_time'][1:].reset_index(drop=True)).reset_index(drop=True)
        df['is_sameDay'] = (lastEnd.dt.day == curStart.dt.day) *1
        Dict[user] = df
    return Dict  
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
# =================================================================================================================================================
'''
# Convert userTrips to a more appropriate foramt for time series forcasting 
'''
def convertToTimeSeries(filenames, userTrips):
    '''
    # Extract the list of columns and define two new lists 
    - One list will be used to define a df for starts 
    - One list will be used to define a df for ends
    - The two dataframes will then be interleaved to from a more proper "time-series"
    '''
    Dict = {}
    for user in filenames: 
        clm_list       = userTrips[user].columns.to_list()
        start_clm_list = [clm for clm in clm_list if 'end_' not in clm]
        end_clm_list   = [clm for clm in clm_list if 'start_' not in clm]
        
        # Form df of starts
        start_df = userTrips[user][start_clm_list]
        start_df['tripDur_min'] = 0
        start_df['tripDis_Km'] = 0
        start_df.rename(columns = {'start_time' : 'time', 'start_location_lat' : 'location_lat', 'start_location_lon' : 'location_lon',\
                                   'start_location_geohash': 'location_geohash', 'start_location_freq': 'location_freq', 'start_class': 'class'} , inplace=True)
        # Form df of ends
        end_df   = userTrips[user][end_clm_list]
        end_df['endToStartDur_min'] = 0
        end_df['endToStartDis_Km'] = 0
        end_df.rename(columns = {'end_time' : 'time', 'end_location_lat' : 'location_lat', 'end_location_lon' : 'location_lon',\
                                   'end_location_geohash': 'location_geohash', 'end_location_freq': 'location_freq', 'end_class': 'class'}  , inplace=True)
        # Form unified df of starts and ends 
        df = pd.concat([start_df, end_df]).sort_values(by='time').reset_index(drop=True)
        # Save the user df in the userTrips dictionary 
        Dict[user] = df
    return Dict 
# =================================================================================================================================================
'''
Unify lat,lon values based on common geohashes 
'''           
def unifyLatLonTimeSeries(filenames, userTrips):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Unifies all lat,lon values based on shared geohashes 
        - Replaces all with the average of all
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable    
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        # All lat,lon within the same geohash will be set to their average
        df['location_lat'] = df['location_lat'].groupby(df['location_geohash']).transform('mean')
        df['location_lon'] = df['location_lon'].groupby(df['location_geohash']).transform('mean')
        
        Dict[user] = df
    return Dict 
# =================================================================================================================================================

'''
# Add geohash ranks based on frequency of occurance  
'''
def addGeoRank(filenames, userTrips):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Adds geohash ranks based on frequency of occurance 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''
    Dict= {}
    for user in filenames: 
        last_month = int(userTrips[user]['time'].iloc[-1].date().strftime('%m')) 
        
        # Define the train df
        train = userTrips[user][userTrips[user]['time'].dt.month < last_month]
        # Set geohash ranks for the train df 
        train['location_freq'] = train.groupby(['location_geohash'])['location_geohash'].transform('count')
        # Get unique geohash keys for the train df
        keys = train['location_geohash'].unique()
        
        # Define the validation df 
        valid = userTrips[user][userTrips[user]['time'].dt.month >= last_month]
        # Default value is 0
        valid['location_freq'] = 0 
        # Geohashes in the validation set that are found in the training set will be set to their values in the training set 
        for i in range(len(keys)):
            if keys[i] in valid['location_geohash'].values:
                valid.loc[valid['location_geohash']==keys[i] , 'location_freq'] = train.loc[train['location_geohash'] == keys[i] , 'location_freq'].values[0] 
        Dict[user] = train.append(valid).reset_index(drop=True)
    return Dict 
# =================================================================================================================================================
'''
# Add distance to top geos
'''
# =================================================================================================================================================
def addDisToTopGeos(filenames, userTrips, n_most_freq = 1):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Converts geohashes to numbers representing various classes and adds the classes to the user dataframe 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].copy(deep=True)
        top_geo_list  = df.sort_values(by='location_freq', ascending=False)['location_geohash'].unique()[:n_most_freq]
        
        for i in range(n_most_freq):
            top = top_geo_list[i]       
            top_lat = df[df['location_geohash'] == top][:1]['location_lat'].values[0]
            top_lon = df[df['location_geohash'] == top][:1]['location_lon'].values[0]
            df[f'dis_to_geo_{i+1}_Km'] = distanceGreatCircle(df['location_lat'], df['location_lon'] , top_lat, top_lon).astype('float16')            
            Dict[user] = df
    return Dict   
# =================================================================================================================================================
'''
# Convert geohashes to targets
'''
def addGeohashClasses(filenames, userTrips):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Converts geohashes to numbers representing various classes and adds the classes to the user dataframe 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable 
    '''
    Dict = {}
    for user in filenames:
        df = userTrips[user].sort_values(by='dis_to_geo_1_Km')
        catList = df['location_geohash'].unique()
        mapping = {key:val for key,val in zip(catList,range(len(catList)))}
        encoder = ce.ordinal.OrdinalEncoder(mapping = [{'col': 'location_geohash', 'mapping': mapping}])
        targets = encoder.fit_transform(df)['location_geohash']
        df.loc[:,'target'] = targets.astype('int32')
        Dict[user] = df.sort_values(by='time')
    return Dict  

# =================================================================================================================================================
'''
# Add datatime features 
'''
def addDatetimeFeatures(filenames, userTrips):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Adds datetime feature to the user dataframe 
    - Returns 
        - A new dictionary for userTrips | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable      
    '''
    us_holidays     = holidays.CountryHoliday('US')
    Dict = {}
    for user in filenames:
        df = userTrips[user]
        # hour_of_day  ----
        df['hour_of_day'] = df['start_time'].dt.hour.astype('int32')
        # Night
        df['night'] =  ((0<=df['hour_of_day']) & (df['hour_of_day']<6)).astype('int32')
        # Morning
        df['morning'] =  ((6<= df['hour_of_day']) & (df['hour_of_day']<12)).astype('int32')
        # Afternoon
        df['afternoon'] =  ( (12<= df['hour_of_day']) & (df['hour_of_day']<18)).astype('int32')
        # Evening 
        df['evening'] =  ( (18<= df['hour_of_day']) & (df['hour_of_day']<24)).astype('int32')        
        # day_of_week  
        df['day_of_week'] = df['start_time'].dt.dayofweek.astype('int32')
        # is_weekend  
        df['is_weekend'] =  (df['start_time'].dt.dayofweek>=5).astype('int32')
        # is_holiday 
        isholiday = pd.Series([1 if date in us_holidays else 0 for date in df['start_time']  ])
        df['is_holiday'] = isholiday.astype('int32')
        # day_of_month
        df['day_of_month'] = df['start_time'].dt.day.astype('int32')
        # month_of_year           
        df['month_of_year'] = df['start_time'].dt.month.astype('int32')
        # ----------------
        # week_of_year
            # Primarily used for grouping and gathering statistics for the average number of trips per week
        df['day_of_year'] = df['start_time'].dt.dayofyear.astype('int32')
        # day_of_year 
            # Primarily used for grouping and gathering statistics for the average number of trips per day
        df['week_of_year'] = df['start_time'].dt.weekofyear.astype('int32')
        
        
        Dict[user] = df
    return Dict
# =================================================================================================================================================
'''
# Provide train/valid datasets 
'''
def train_val_split(filenames, userTrips, lastmonth):
    '''
    - Takes a list of user filenames and a userTrips dictionary 
    - Uses the last month parameter (int) for validation and train/valid split
    - Returns 
        - Two new dictionaries for userTrips: one for training and one for validation | type: dict
        - For typical workflow, overwrite the userTrips dictionary with this one
        - For diagnostic, comparison and benchmark pusposes, save the new dataframe in a different variable    
    '''
    userTrains = {}
    userValids = {}
    for user in filenames:
        userTrains[user] = userTrips[user][userTrips[user]['time'].dt.month < lastmonth ] 
        userValids[user] = userTrips[user][userTrips[user]['time'].dt.month == lastmonth ] 
    return userTrains, userValids
# =================================================================================================================================================
