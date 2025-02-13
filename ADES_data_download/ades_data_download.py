
        
        
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import os
import requests
import time


shape = gpd.read_file("..../1976_final_classes.shp")

# Function to get Hubeau URL
def getHubeauURL_piezo(code_bss, date_i, date_f):
    url_root = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques?"
    url_code = "code_bss=" + code_bss.replace("/", "%2F")
    url_date = "&date_debut_mesure=" + date_i + "&date_fin_mesure=" + date_f
    url_tail = "&size=20000&sort=asc"
    return url_root + url_code + url_date + url_tail

# Function to convert piezo data to pandas DataFrame
# def piezoDico_to_pandas_df(res):
#     df = pd.DataFrame.from_dict(res["data"])
#     print("DataFrame columns:", df.columns)
    
#     # Check if the columns 'date_mesure' and 'niveau_nappe_eau' exist in the data
#     if 'date_mesure' in df.columns and 'niveau_nappe_eau' in df.columns:
#         # Rename columns for consistency
#         df.rename(columns={'date_mesure': 'date', 'niveau_nappe_eau': 'hydraulic_head'}, inplace=True)
#         df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
#         return df
#     else:
#         print("Required columns not found in the data.")
#         return None
def piezoDico_to_pandas_df(res, code_bss):
    df = pd.DataFrame.from_dict(res["data"])
    
    # Check if the columns 'date_mesure' and 'val_calc_ngf' exist in the data
    if 'date_mesure' in df.columns and 'niveau_nappe_eau' in df.columns:
        # Select only the necessary columns and rename them
        df = df[['code_bss', 'date_mesure', 'niveau_nappe_eau', 'qualification']]
        df.columns = ['code', 'date', 'hydraulic_head', 'qualification']
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        return df
    else:
        print(f"Required columns not found in the data for code {code_bss}. Skipping.")
        return None


# Definition of a period of interest:
mydate_i = "1900-01-01"  # year/month/day
mydate_f = "2024-11-30"  # year/month/day

# Define the directory path to save data
save_directory = './'

# Create an empty DataFrame to store all data
all_data = pd.DataFrame(columns=['code', 'date', 'hydraulic_head', 'qualification'])

for code in shape['code_bss'].unique():
    print(code)
    myurl = getHubeauURL_piezo(code, mydate_i, mydate_f)
    r = requests.get(myurl)
    res = r.json()

    # Check if the "data" key exists and if there's data available
    if "data" in res and len(res["data"]) > 0:
        df = piezoDico_to_pandas_df(res, code)  # Pass 'code' as an argument
        
        if df is not None:
            print(f"Processing code: {code}")

            # Append the data to the all_data DataFrame
            all_data = pd.concat([all_data, df[['code', 'date', 'hydraulic_head', 'qualification']]], ignore_index=True)

    else:
        print(f"No data found for code {code}.")

# Save the combined data to a CSV file
all_data.to_csv(os.path.join(save_directory, 'new_data_all.csv'), index=False)

