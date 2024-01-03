# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Define some variables
parse = lambda x: datetime.fromtimestamp(float(x)/1000)

path = 'nature/sms-call-internet-mi-'
scaler = MinMaxScaler(feature_range=(-1, 1))
#----------------------------------------------------------------------------#
# Call from main.py
#----------------------------------------------------------------------------#
def load_data():
    
    dates = get_dates(2013, 11, 4)
    train_data, train_targets = load_target_data(0,40, dates)
    test_data, test_targets = load_target_data(40,49, dates) 
    val_data, val_targets = load_target_data(49,56, dates)
    
    return (train_data, train_targets), (test_data, test_targets),(val_data, val_targets), scaler
    

#----------------------------------------------------------------------------#
# Load data from files, create target and data
#----------------------------------------------------------------------------#

def load_target_data(start_day, end_day, dates):
    
    data_concat = []
    
    # Read files
    for i in range(start_day, end_day):
        
        data = pd.read_csv(path + str(dates[i]) +'.txt', sep='\t', encoding="utf-8-sig", names=['CellID', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin', 'callout', 'internet'], dtype={'datetime': 'object'})
        
        # Fix datetime column
        data = transform_datetime(data)        
        
        # Filter with 1000 cells
        data = filter_traffic(data)
        
        data_concat.append(data) 
    
    # Concat all files    
    data = pd.concat(data_concat, ignore_index=True) 
    
    name = 'data'+str(start_day)+str(end_day)+'.csv'
    #Uncomment when data is saved
    #data = pd.read_csv(name)
    # Save data first time you read it
    data.to_csv(name, index=False)
    
    # Create data and targets
    targets = pd.DataFrame()
    targets['call'] = data['callin'] + data['callout']
    data ['sms'] = data['smsin'] + data['smsout']
    data = data.drop(['callin','callout', 'countrycode','smsin','smsout'], axis = 1)
    
    # Normalize continuous data
    data[['sms','internet']] = scaler.fit_transform(data[['sms','internet']])  
    targets['call'] = scaler.fit_transform(targets[['call']])
  
    # Transform series and df into float array
    data = np.array(data, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
     
        
    return data, targets


#----------------------------------------------------------------------------#
# Modify read file
#----------------------------------------------------------------------------# 
def transform_datetime(data):
    
    # Transform ms in seconds
    data['datetime'] = data['datetime'].apply(parse)
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    data = data.set_index('datetime')
    data['hour'] = data.index.hour
    data['weekday'] = data.index.weekday
    data = data.groupby(['hour', 'weekday', 'CellID'], as_index=False).sum()
    
    return data
#----------------------------------------------------------------------------#
# Filter top cells read file
#----------------------------------------------------------------------------#
def filter_traffic(data):
    
    #Filter top 100 cells from each day
    top_cells_df = pd.read_csv('top_5000_cells.csv')
    top_cells_overall = top_cells_df['CellID'].tolist()
    filtered_data = data[data['CellID'].isin(top_cells_overall)]
    
    return filtered_data

#----------------------------------------------------------------------------#
# Get dates and make a list
#----------------------------------------------------------------------------#
def get_dates(year, month, day):
    
    # Get start date
    start_date = datetime(year, month, day)

    # Number of days
    num_days = 56

    # Create date list
    date_list = [start_date + timedelta(days=i) for i in range(num_days)]

    # Format dates
    formatted_dates = [date.strftime('%Y-%m-%d') for date in date_list]
    
    return formatted_dates