import numpy as np
import pandas as pd

def get_data():
   
    df = pd.read_csv("housing.csv")

    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

   
    df = df.drop('ocean_proximity', axis=1)

    
    Y = df['median_house_value'].values
    X = df.drop('median_house_value', axis=1).values 
    return X, Y


 