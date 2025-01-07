modtype="BILSTM" #GRU, LSTM,BILSTM
seq_length=48
initm=10
test_size=0.2
GWL_type= "all FRANCE" #annual, inertial, mixed

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.keras.backend.get_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD
from optuna.integration import TFKerasPruningCallback
import optuna
from uncertainties import unumpy
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from typing import Tuple


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*3)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

ds = xr.open_dataset('/media/chidesiv/DATA2/Phase2/3_stations/ERA5_inputdata/1940_2022/ERA5_1940_2022.nc')


def calculate_rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse



def calculate_nrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    data_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / data_range
    return nrmse


def calculate_nse(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse



# import cfgrib

# # Open the GRIB file with the filter_by_keys option
# ds = cfgrib.open_dataset('/home/chidesiv/Desktop/Phase2/3_stations/ERA5_inputdata/ERA5_1950_2020_9KM.grib', engine='cfgrib', filter_by_keys={'stepType': 'avgid'})

# Now you can work with the opened dataset as you would with a NetCDF file


def Scaling_data(X_train,X_valid,y_train):
    """
    Scaling function to fit on training data and transform on test data

    Parameters
    ----------
    X_train : TYPE
        Original training data of input variables.
    X_valid : TYPE
        Original testing data.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_s : TYPE
        scaled training input variables
    X_valid_s : TYPE
        scaled test input variables
    y_train_s : TYPE
        scaled training target variable.
    X_C_scaled : TYPE
        combined scaled training and test input variables

    """
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(X_train)
    X_valid_s=Scaler.transform(X_valid)
    target_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler.fit(np.array(y_train).reshape(-1,1))
    y_train_s=target_scaler.transform(np.array(y_train).reshape(-1,1))
    
    X_C_scaled=np.concatenate((X_train_s,X_valid_s),axis=0)
   
    return X_train_s,X_valid_s,y_train_s,X_C_scaled
 
def Scaling_extend(X_train,X_extend):
    """
    Scaling function to fit on training data and transform on test data

    Parameters
    ----------
    X_train : TYPE
        Original training data of input variables.
    X_valid : TYPE
        Original testing data.
    y_train : TYPE
        DESCRIPTION.

    Returns
    -------
    X_train_s : TYPE
        scaled training input variables
    X_valid_s : TYPE
        scaled test input variables
    y_train_s : TYPE
        scaled training target variable.
    X_C_scaled : TYPE
        combined scaled training and test input variables

    """
    Scaler=MinMaxScaler(feature_range=(0,1))
    X_train_s=Scaler.fit_transform(X_train)
    X_extend_s=Scaler.transform(X_extend)
    
    
    
   
    return X_extend_s
 

def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

def reshape_onlyinputdata(x: np.ndarray,  seq_length: int) -> Tuple[np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        

    return x_new


#Hyperparameter tuning with Optuna for LSTM


def func_dl(trial):
   with device('/gpu:0'):    
    #     tf.config.experimental.set_memory_growth('/gpu:0', True)    
        set_seed(2)
        seed(1)
        
        
        
        optimizer_candidates={
            "adam":Adam(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "SGD":SGD(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True)),
            # "RMSprop":RMSprop(learning_rate=trial.suggest_float('learning_rate',1e-3,1e-2,log=True))
        }
        optimizer_name=trial.suggest_categorical("optimizer",list(optimizer_candidates))
        optimizer1=optimizer_candidates[optimizer_name]

    
        callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights = True),TFKerasPruningCallback(trial, monitor='val_loss')]
        
        epochs=trial.suggest_int('epochs', 50, 500,step=50)
        batch_size=trial.suggest_int('batch_size', 16,256,step=16)
        #weight=trial.suggest_float("weight", 1, 5)
        n_layers = trial.suggest_int('n_layers', 1, 1)
        model = Sequential()
        for i in range(n_layers):
            num_hidden = trial.suggest_int("n_units_l{}".format(i), 10, 100,step=10)
            return_sequences = True
            if i == n_layers-1:
                return_sequences = False
            # Activation function for the hidden layer
            if modtype == "GRU":
                model.add(GRU(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences))
            elif modtype == "LSTM":
                model.add(LSTM(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences))
            elif modtype == "BILSTM":
                model.add(Bidirectional(LSTM(num_hidden,input_shape=(X_train_l_all.shape[1],X_train_l_all.shape[2]),return_sequences=return_sequences)))
            model.add(Dropout(trial.suggest_float("dropout_l{}".format(i), 0.2, 0.2), name = "dropout_l{}".format(i)))
            #model.add(Dense(units = 1, name="dense_2", kernel_initializer=trial.suggest_categorical("kernel_initializer",['uniform', 'lecun_uniform']),  activation = 'Relu'))
        #model.add(BatchNormalization())study_blstm_
        model.add(Dense(1))
        model.compile(optimizer=optimizer1,loss="mse",metrics=['mse'])
        ##model.summary()


        model.fit(X_train_l_all,y_train_l_all,validation_data = (X_valid_ls,y_valid_ls ),shuffle = False,batch_size =batch_size,epochs=epochs,callbacks=callbacks
                  ,verbose = False)
  
        score=model.evaluate(X_valid_ls,y_valid_ls)

       

        return score[1]



def optimizer_1(learning_rate,optimizer):
        tf.random.set_seed(init+11111)
        if optimizer==Adam:
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer==SGD:
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        return opt


def gwlmodel(init, params,X_train_l, y_train_l, modtype, GWL_type):
        with tf.device('/gpu:0'):
            seed(init+99999)
            tf.random.set_seed(init+11111)
            par= params.best_params
            # seq_length = par.get(par_names[0])
            learning_rate=par.get(par_names[0])
            optimizer = par.get(par_names[1])
            epochs = par.get(par_names[2])
            batch_size = par.get(par_names[3])
            n_layers = par.get(par_names[4])
            # X_train, X_valid, y_train, y_valid=Scaling_data( X_tr'+ str(file_c['code_bss'][k])], X_va'+ str(file_c['code_bss'][k])], y_t'+ str(file_c['code_bss'][k])], y_v'+ str(file_c['code_bss'][k])])
            # X_train_l, y_train_l = reshape_data(X_train, y_train,seq_length=seq_length)
            model = Sequential()
            # i = 1
            for i in range(n_layers):
                return_sequences = True
                if i == n_layers-1:
                    return_sequences = False
                if modtype == "GRU":
                    model.add(GRU(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "LSTM":
                    model.add(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences))
                elif modtype == "BILSTM":
                    model.add(Bidirectional(LSTM(par["n_units_l{}".format(i)],input_shape=(X_train_l.shape[1],X_train_l.shape[2]),return_sequences=return_sequences)))
                model.add(Dropout(par["dropout_l{}".format(i)]))
            model.add(Dense(1))
            opt = optimizer_1(learning_rate,optimizer)
            model.compile(optimizer = opt, loss="mse",metrics = ['mse'])
            callbacks = [EarlyStopping(monitor = 'val_loss', mode ='min', verbose = 1, patience=50,restore_best_weights = True), tf.keras.callbacks.ModelCheckpoint(filepath='best_model'+str(modtype)+str(GWL_type)+str(init)+'.h5', monitor='val_loss', save_best_only = True, mode = 'min')]
            model.fit(X_train_l, y_train_l,validation_split = 0.2, batch_size=batch_size, epochs=epochs,callbacks=callbacks)
        return model
    


# read the shapefile


shape = gpd.read_file("/media/chidesiv/DATA2/Phase2/QGIS/classify/final/1976_final_classes.shp")

# extract the coordinates
coords = shape.geometry.apply(lambda x: x.coords[:]).apply(pd.Series)



coords['code_bss'] = shape['code_bss']
coords['Class'] = shape['Class']
# rename the columns
coords.columns = ['x','code_bss','Class']


# convert the DataFrame to two columns
coords[['long','lat']] = pd.DataFrame(coords.x.tolist(), index=coords.index)

# drop the original column
coords.drop(columns=['x'], inplace=True)


from pyproj import Transformer
transformer = Transformer.from_crs( "EPSG:2154","EPSG:4326")
x, y = coords['lat'], coords['long']
#coords['latdeg'], coords['longdeg']= transformer.transform(x, y)

# Create an empty dictionary to store grid values for each code_bss
grid_values_dict = {}

# Iterate over each code_bss in the dataframe
for code_bss in coords['code_bss'].unique():
    # Filter the dataframe to get rows with the current code_bss
    df_code_bss = coords[coords['code_bss'] == code_bss]
    # Extract the grid values for each latdeg and longdeg
    grid_values = ds.sel(latitude=df_code_bss['lat'].values, longitude=df_code_bss['long'].values, method='nearest')
    # Add the grid values to the dictionary
    grid_values_dict[code_bss] = grid_values

# You can now access the grid values for each code_bss in the grid_values_dict
# print(grid_values_dict)


result = {}
for code in grid_values_dict.keys():
    df = pd.DataFrame()
    
    for var in grid_values_dict[code].data_vars:
        # print(var)
        # df_var = pd.DataFrame(grid_values_dict[code][var].values)
        df_var = grid_values_dict[code][var].to_dataframe()
        df_var.columns = [var]
        df = pd.concat([df, df_var], axis=1)
    df["code_bss"] = code
    result[code] = df

df_result1 = pd.concat(result.values())
# df_result = df_result.set_index(['time','expver'])
# df_result = df_result.query('expver == 1')
df_result= df_result1.drop(columns=['d2m','skt'])

df_result = df_result.dropna()


file_GWL =pd.read_csv(r"/media/chidesiv/DATA2/National_scale_models/newdata/new_data_all.csv")


file_GWL = file_GWL.rename(columns={'code': 'code_bss', 'hydraulic_head': 'val_calc_ngf','date':'date_mesure'})
# file_GWL =pd.read_csv(r"/media/chidesiv/DATA2/PhD/data/BRGM_DATA/Pour_Siva/Pour_Siva/GWL_time_series/chroniques_pretraitees.txt", sep='$')
file_GWL['code_bss'].unique()
file_GWL['date_mesure'] = pd.to_datetime(file_GWL['date_mesure'])

df_list = []
for code in grid_values_dict.keys():
    df = file_GWL[file_GWL['code_bss'] == code]
    df = df.assign(date=df['date_mesure']).drop(columns=['date_mesure'])
    df_list.append(df)

gwl_combine = pd.concat(df_list)


#Combined available GWL data with input data

gwl_combine['date'] = pd.to_datetime(gwl_combine['date'])
gwl_combine = gwl_combine.set_index('date')
gwl_combine = gwl_combine.groupby('code_bss').resample('MS').mean()

gwl_combine = gwl_combine.interpolate(method='linear')
gwl_combine = gwl_combine.reset_index()

df_result = df_result.rename(columns={'code_bss': 'code_bss'})
df_result['date'] = pd.to_datetime(df_result.index.get_level_values(0))
df_result = df_result.set_index('date')
# df_result = df_result.groupby('code_bss').resample('MS').mean(numeric_only=True)
df_result = df_result.reset_index()

final_df = pd.merge(df_result, gwl_combine, on=['code_bss','date'], how='inner')


data_static= pd.read_csv(r'/media/chidesiv/DATA2/Phase2/3_stations/ERA5_inputdata/bss_with_static_variables_corrected.csv')



merged_df_static = pd.merge(df_result, data_static[['code_bss', 'longitude', 'latitude', 'etateh', 'milieueh', 'cdlitho']], on='code_bss')



import pandas as pd

# Select the columns for one-hot encoding
columns_to_encode = ['etateh', 'milieueh', 'cdlitho']

# Apply one-hot encoding using get_dummies()
encoded_df = pd.get_dummies(merged_df_static, columns=columns_to_encode)

# Retrieve the one-hot encoded 'code_bss' column
encoded_code_bss = pd.get_dummies(merged_df_static['code_bss'], prefix='code_bss')

# Merge the one-hot encoded 'code_bss' column with the rest of the encoded DataFrame
encoded_df = pd.concat([encoded_df, encoded_code_bss], axis=1)

# Print the encoded dataframe
print(encoded_df)




# one_hot = pd.get_dummies(df_result['code_bss'])

# final_df1  = pd.concat([df_result, one_hot], axis=1)
# Extract the period of input data available for reconstruction based on first date of GWL available 



# Create dictionary to store extended dataframes
# extended_dfs = {}

# X_train_l_all = np.empty((0, seq_length, encoded_df.shape[1]-2))
# y_train_l_all = np.empty((0,  1))


X_train_l_all = np.empty((0, seq_length, encoded_df.shape[1]-2))
y_train_l_all = np.empty((0,  1))

X_valid_ls= np.empty((0, seq_length, encoded_df.shape[1]-2))
y_valid_ls= np.empty((0,  1))

# models = {}
# for init in range(initm):
#     # models[init] = gwlmodel(init, Study_DL, X_train_l_all, y_train_l_all, modtype, GWL_type)
#     models[init] = tf.keras.models.load_model('best_model'+str(modtype)+str(GWL_type)+str(init)+'.h5') 

# importance_dfs = []
# Create dictionary to store extended dataframes
# extended_dfs = {}

# Iterate over each code_bss
for code in final_df['code_bss'].unique():
    # Extract data for the current code_bss from final_df
    #Extract Class value for each code_bss from coords dataframe
    Class = coords[coords['code_bss'] == code]['Class'].values[0]

    df = encoded_df[encoded_df['code_bss'] == code]
    df_x1=final_df[final_df['code_bss'] == code]
    df_x=df_x1.drop(columns=['code_bss'])
    # Split into X and y
    X = df.drop(columns=['code_bss'])

    y = df_x1[['val_calc_ngf']]
    
    minimum_date= pd.to_datetime('1970-01-01')


    min_train_date = max(minimum_date, df_x1['date'].min())
    # min_train_date = pd.to_datetime('1969-12-01')#X['date'].min()
    # lev=4 # number of levels for decomposition, here we choose 4
    
    
    # f=((int(pow(2, lev))-1)*(L-1))+1 # number of boundary affected coefficients to be removed as per quilty and adamowski (2018)
    # Add 48 months to the minimum date
    
    # Add 48 months to the minimum date
    first_date = min_train_date  - pd.DateOffset(months=48)
    # Filter X and y for train and test sets
    start_date = '2015-01-01'
    end_date = '2023-08-01'
    df_x.reset_index(drop=True, inplace=True)
    # X.reset_index(drop=True, inplace=True)
    # y.reset_index(drop=True, inplace=True)
    
    X_train = X[(df['date'] > first_date) & (df['date'] <= '2014-12-01')]
    print(X_train.shape)
    y_train = y[(df_x1['date'] >= min_train_date) & (df_x1['date'] <= '2014-12-01')]
    X_test = X[(X['date'] >= start_date) & (X['date'] <= end_date)]
    y_test = y[(df_x1['date'] >= start_date) & (df_x1['date'] <= end_date)]
    # y_train.plot()
    
    # Extract rows from df_result with dates before the first date in X_train
    # Calculate the minimum date in the 'date' column
    # min_date = X['date'].min()
   
    # extended_df = df_result[(df_result['code_bss'] == code) & (df_result['date'] < min_date)]
    
    # # Store the extended dataframe in the dictionary
    # extended_dfs[code] = extended_df
    
    # # Create X_extend and y_extend by filtering the extended_df
    # X_extend = extended_df[extended_df['date'] < min_date].drop(columns=['code_bss'])
    # # print(f'code_bss: {code}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
    X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data(X_train.drop(columns=['date']), X_test.drop(columns=['date']), y_train)
    
    # X_extend_s=Scaling_extend(X_train.drop(columns=['date']), X_extend.drop(columns=['date']))
    
    # X_train_l,y_train_l =reshape_data(X_train_s,y_train_s,seq_length=seq_length)
    X_train_l=reshape_onlyinputdata(X_train_s,seq_length=seq_length)

    y_train_l=y_train_s

    print(X_train_l.shape)
    print(y_train_l.shape)
    # Concatenate X_train_l and y_train_l with the previous code_bss
    X_train_, X_valid_, y_train_, y_valid_  = train_test_split(X_train_l, y_train_l , test_size=0.2,random_state=1,shuffle=False)
        
    
    
    X_valid_ls = np.concatenate([X_valid_ls, X_valid_], axis=0)
    y_valid_ls = np.concatenate([y_valid_ls, y_valid_], axis=0)
    # Concatenate X_train_l and y_train_l with the previous code_bss
    X_train_l_all = np.concatenate([X_train_l_all, X_train_l], axis=0)
    y_train_l_all = np.concatenate([y_train_l_all, y_train_l], axis=0)
    print(X_train_l_all.shape)
    print(y_train_l_all.shape)
    X_c= reshape_onlyinputdata(X_C_scaled,seq_length=seq_length)
    # X_extend_l= reshape_onlyinputdata(X_extend_s,seq_length=seq_length)
    X_valid_l=X_c[int((len(X_train_l))):]


import pickle
with open('/media/chidesiv/DATA2/Gauged_simplified/1_layer/Multi_station/Static_OHE/Standalone/'+'study_'+str(modtype)+str(GWL_type)+'.pkl','rb') as f:
  Study_DL=pickle.load(f)

print(modtype)
par = Study_DL.best_params
par_names = list(par.keys())
par

   
    feature_names = X_train.columns[1:]
    for init in range(initm):
        # model = tf.keras.models.load_model('./'+'best_model'+str(k)+str(init)+'.h5')
        # background = 
        # globals()["e"+str(init)] = shap.DeepExplainer((globals()["model"+str(init)].layers[0].input,globals()["model"+str(init)].layers[-1].output), globals()['X_train_lf'+str(file_c['code_bss'][k])])
        globals()["e"+str(init)] = shap.DeepExplainer((models[init]), X_train_l)
        globals()["shap_values"+str(init)] = globals()["e"+str(init)].shap_values(X_valid_l)
       
        
        shap_vals = np.asarray(globals()["shap_values"+str(init)][0])
        shap_vals = shap_vals.reshape(-1, shap_vals.shape[-1])
        
        
        if init == 0:
                 s = shap_vals
                 X = X_valid_l.reshape(-1, X_valid_l.shape[-1])
        else:
                 s = np.append(s,shap_vals,0)
                 X = np.append(X,X_valid_l.reshape(-1, X_valid_l.shape[-1]),0)
                 
    cohorts = {"": s}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = np.abs(cohort_exps[i]).mean(0)
    features = cohort_exps[0].data
    # feature_names = cohort_exps[0].feature_names
    values = np.array(cohort_exps)

    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    top_variables = feature_importance['features'].values[:X_train.columns[1:].shape[0]]
    top_variable_names = ['Code'] +['Class'] + [f'{i}th important variable' for i in range(1, X_train.columns[1:].shape[0]+1)]
    top_variable_values = [code] +[Class] + top_variables.tolist()
    top_variables_dict = dict(zip(top_variable_names, top_variable_values))
    # all_feature_importances = all_feature_importances.append(top_variables_dict, ignore_index=True)

    importance_dfs.append(pd.DataFrame(top_variables_dict, index=[0]))
        # np.savetxt('./'+
        # shap.summary_plot( globals()["shap_values"+str(init)][0], plot_type = 'bar', feature_names = globals()['X_tr'+str(file_c['code_bss'][k])].columns)
    plt.figure()

    shap.summary_plot(s, X,feature_names = feature_names, show=False,color_bar=False)              
    plt.colorbar()
    plt.savefig('./'+'SHAP'+str(modtype)+str(GWL_type)+str(initm)+'{}.png'.format(code.replace('/','_')), dpi=300, bbox_inches = 'tight', pad_inches = 0.1)    


all_feature_importances = pd.concat(importance_dfs, ignore_index=True) 
all_feature_importances.to_csv('imp_variables'+str(modtype)+str(GWL_type)+'.csv', index=False)
# X_train_ls, X_valid_ls, y_train_ls, y_valid_ls  = train_test_split(X_train_l_all, y_train_l_all , test_size=0.2,random_state=1,shuffle=False)
        
        
        
# Study_DL= optuna.create_study(direction='minimize',sampler=TPESampler(seed=10),study_name='study_'+str(modtype)+str(GWL_type))
# Study_DL.optimize(func_dl,n_trials=30) 
# # #         # file_name = f"{code.replace('/','_')}.pkl"
# # #         # with open(file_name, 'wb') as file:
# # #         #     pickle.dump(Study_DL, file)
# pickle.dump(Study_DL,open('./'+'study_'+str(modtype)+str(GWL_type)+'.pkl', 'wb'))
# # with open('study_'+str(modtype)+str(GWL_type)+'.pkl','rb') as f:
# #     Study_DL = pickle.load(f)   
# par = Study_DL.best_params
# par_names = list(par.keys())
   
# models = {}
# for init in range(initm):
#     models[init] = gwlmodel(init, Study_DL, X_train_l_all, y_train_l_all, modtype, GWL_type)
#     # models[init] = tf.keras.models.load_model('best_model'+str(modtype)+str(GWL_type)+str(init)+'.h5')
# # Create dictionary to store extended dataframes
# # extended_dfs = {}
# scores = pd.DataFrame(columns=['code_bss','Class',
#                         'NSE_tr',
#                         'KGE_tr',
#                         'MAE_tr',
#                         'RMSE_tr',
#                         'NRMSE_tr',
#                         'NSE_te',
#                         'KGE_te',
#                         'MAE_te',
#                         'RMSE_te',
#                         'NRMSE_te'])
# # Iterate over each code_bss
# # Iterate over each code_bss


# target_scalers = {}
# for code in final_df['code_bss'].unique():
#     # Extract data for the current code_bss from final_df
#     #Extract Class value for each code_bss from coords dataframe
#     Class = coords[coords['code_bss'] == code]['Class'].values[0]

#     df = encoded_df[encoded_df['code_bss'] == code]
#     df_x1=final_df[final_df['code_bss'] == code]
#     df_x=df_x1.drop(columns=['code_bss'])
#     # Split into X and y
#     X = df.drop(columns=['code_bss'])

#     y = df_x1[['val_calc_ngf']]
    
#     minimum_date= pd.to_datetime('1970-01-01')


#     min_train_date = max(minimum_date, df_x1['date'].min())
#     # min_train_date = pd.to_datetime('1969-12-01')#X['date'].min()
#     # lev=4 # number of levels for decomposition, here we choose 4
    
    
#     # f=((int(pow(2, lev))-1)*(L-1))+1 # number of boundary affected coefficients to be removed as per quilty and adamowski (2018)
#     # Add 48 months to the minimum date
    
#     # Add 48 months to the minimum date
#     first_date = min_train_date  - pd.DateOffset(months=48)
#     # Filter X and y for train and test sets
#     start_date = '2015-01-01'
   

#     # Convert the string to a Timestamp object
#     end_date = min(pd.to_datetime('2023-08-01'), file_GWL[file_GWL['code_bss'] == code]['date_mesure'].max())
    
#     df_x.reset_index(drop=True, inplace=True)
#     # X.reset_index(drop=True, inplace=True)
#     # y.reset_index(drop=True, inplace=True)
    
#     X_train = X[(df['date'] > first_date) & (df['date'] <= '2014-12-01')]
#     print(X_train.shape)
#     y_train = y[(df_x1['date'] >= min_train_date) & (df_x1['date'] <= '2014-12-01')]
#     X_test = X[(X['date'] >= start_date) & (X['date'] <= end_date)]
#     y_test = y[(df_x1['date'] >= start_date) & (df_x1['date'] <= end_date)]
    
#     # Extract rows from df_result with dates before the first date in X_train
#     # Calculate the minimum date in the 'date' column
#     # min_date = X['date'].min()
   
#     # extended_df = df_result[(df_result['code_bss'] == code) & (df_result['date'] < min_date)]
#     # extended_df = df_result[(df_result['code_bss'] == code) & (df_result['date'] <= min_train_date)]
    
#     # Store the extended dataframe in the dictionary
#     # extended_dfs[code] = extended_df
    
#     # Create X_extend and y_extend by filtering the extended_df
#     # X_extend = extended_df[extended_df['date']<= min_train_date].drop(columns=['code_bss'])
#     # Store the extended dataframe in the dictionary
#     # extended_dfs[code] = extended_df
    
#     # Create X_extend and y_extend by filtering the extended_df
#     # X_extend = extended_df[extended_df['date'] < min_date].drop(columns=['code_bss'])
#     # print(f'code_bss: {code}, X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}')
#     X_train_s, X_valid_s, y_train_s,X_C_scaled=Scaling_data(X_train.drop(columns=['date']), X_test.drop(columns=['date']), y_train)
    
#     # X_extend_s=Scaling_extend(X_train.drop(columns=['date']), X_extend.drop(columns=['date']))
    
#     # X_train_l,y_train_l =reshape_data(X_train_s,y_train_s,seq_length=seq_length)
#     X_train_l=reshape_onlyinputdata(X_train_s,seq_length=seq_length)

#     y_train_l=y_train_s
#     # # Concatenate X_train_l and y_train_l with the previous code_bss
#     # X_train_l_all = np.concatenate([X_train_l_all, X_train_l], axis=0)
#     # y_train_l_all = np.concatenate([y_train_l_all, y_train_l], axis=0)
    
#     X_c= reshape_onlyinputdata(X_C_scaled,seq_length=seq_length)
#     # X_extend_l= reshape_onlyinputdata(X_extend_s,seq_length=seq_length)
#     X_valid_l=X_c[int((len(X_train_l))):]

#     target_scalers[code] = MinMaxScaler(feature_range=(0, 1))
#     target_scalers[code].fit(np.array(y_train).reshape(-1, 1))
#     sim_init = np.zeros((len(X_valid_l), initm))
#     sim_init[:] = np.nan
#     sim_tr = np.zeros((len(X_train_l), initm))
#     sim_tr[:] = np.nan
#     # sim_ex = np.zeros((len(X_extend_l), initm))
#     # sim_ex[:] = np.nan
#     # scores_tr = pd.DataFrame(columns=['R2_tr','RMSE_tr','MAE_tr'])
#     # scores_te = pd.DataFrame(columns=['R2_te','RMSE_te','MAE_te'])
    
#     # for code in final_df['code_bss'].unique():    
#     for init in range(initm):
#         # model = gwlmodel(init, Study_DL,X_train_l_all, y_train_l_all, modtype, GWL_type)
#         # model=tf.keras.models.load_model('best_model'+str(modtype)+str(init)+'{}.h5'.format(code.replace('/','_')))
#         y_pred_valid = models[init].predict(X_valid_l)
#         sim_test = target_scalers[code].inverse_transform(y_pred_valid)
#         y_pred_train = models[init].predict(X_train_l)
#         sim_train = target_scalers[code].inverse_transform(y_pred_train)
#         # y_pred_extend = globals()['model'+str(init)].predict(X_extend_l)
#         # sim_extend = globals()['target_scaler'+str(code)].inverse_transform(y_pred_extend)
    
#         sim_init[:, init] = sim_test[:, 0]
#         sim_tr[:, init] = sim_train[:, 0]
#         # sim_ex[:, init] = sim_extend[:, 0]
        
    
#     sim_f = pd.DataFrame(sim_init)
#     sim_t = pd.DataFrame(sim_tr)
#     # sim_e = pd.DataFrame(sim_ex)
#     lower_percentile = 2.5  # 2.5th percentile
#     upper_percentile = 97.5  # 97.5th percentile
    
#     # Calculate the lower and upper bounds of the prediction interval
#     lower_bound_train = np.percentile(sim_tr, lower_percentile, axis=1)
#     upper_bound_train = np.percentile(sim_tr, upper_percentile, axis=1)
    
#     lower_bound_test = np.percentile(sim_init, lower_percentile, axis=1)
#     upper_bound_test = np.percentile(sim_init, upper_percentile, axis=1)
    
#     # lower_bound_extend = np.percentile(sim_ex, lower_percentile, axis=1)
#     # upper_bound_extend = np.percentile(sim_ex, upper_percentile, axis=1)
    
#     sim_median = sim_f.median(axis=1)
#     sim_tr_median = sim_t.median(axis=1)
#     # sim_ex_median=sim_e.median(axis=1)
    
#     # sim_init_uncertainty = unumpy.uarray(sim_f.mean(axis=1), 1.96 * sim_f.std(axis=1))
#     # sim_tr_uncertainty = unumpy.uarray(sim_t.mean(axis=1), 1.96 * sim_t.std(axis=1))
#     # sim_ex_uncertainty = unumpy.uarray(sim_ex.mean(axis=1), 1.96 * sim_ex.std(axis=1))
#     # Calculate the desired percentiles for the prediction interval
    

#     sim = np.asarray(sim_median).reshape(-1, 1)
#     sim_train = np.asarray(sim_tr_median).reshape(-1, 1)
#     # sim_extend=np.asarray(sim_ex_median).reshape(-1, 1)
    
#     obs_tr = np.asarray(target_scalers[code].inverse_transform(y_train_l).reshape(-1, 1))
    
#     # y_err = unumpy.std_devs(sim_init_uncertainty)
#     # y_err_tr = unumpy.std_devs(sim_tr_uncertainty)
#     # y_err_ex= unumpy.std_devs(sim_ex_uncertainty)


#     #Compute NSE, MSE, R2, RMSE, MAE, KGE for training data
#     NSE_tr=calculate_nse(obs_tr, sim_train)
    
#     # Calculate correlation coefficient for training data
#     corr_coef_tr = np.corrcoef(obs_tr, sim_train, rowvar=False)[0, 1]

#     # Calculate standard deviation ratio for training data
#     std_ratio_tr = np.std(sim_train) / np.std(obs_tr)
#     # Calculate mean ratio for training data
#     mean_ratio_tr = np.mean(sim_train) / np.mean(obs_tr)
#     # Calculate KGE for training data
#     KGE_tr = 1 - np.sqrt((corr_coef_tr - 1) ** 2 + (std_ratio_tr - 1) ** 2 + (mean_ratio_tr - 1) ** 2)
   
    
 
#     # Calculate MAE, MSE, R2, and RMSE for training data
#     MAE_tr = mean_absolute_error(obs_tr, sim_train)
#     mse_tr = mean_squared_error(obs_tr, sim_train)
#     R2_tr = r2_score(obs_tr, sim_train)
#     RMSE_tr = math.sqrt(mse_tr)
#     nrmse_tr = calculate_nrmse(obs_tr, sim_train)

    
#     # scores_tr = pd.DataFrame(np.array([[ R2_tr, RMSE_tr, MAE_tr]]),
#     #                    columns=['R2_tr','RMSE_tr','MAE_tr'])
#     #scores_tr = scores_tr.append(pd.DataFrame(np.array([[ R2_tr, RMSE_tr, MAE_tr]]), columns=['R2_tr','RMSE_tr','MAE_tr']), ignore_index=True)
    
#     #scores_tr.to_csv('scores_tr_era5'+str(modtype)+'{}.csv'.format(code.replace('/','_')))
    
#     #Compàute NSE, MSE, R2, RMSE, MAE, KGE for test data
#     NSE_te=calculate_nse(y_test, sim)
#     MAE_te=mean_absolute_error(y_test, sim)
#     mse_te=mean_squared_error(y_test, sim)
#     R2_te=r2_score(y_test, sim)
#     RMSE_te=math.sqrt(mse_te)
#     NRMSE_te = calculate_nrmse(y_test, sim)
     
#     # Calculate correlation coefficient for testing data
#     corr_coef_te = np.corrcoef(y_test, sim_test, rowvar=False)[0, 1]
#     # Calculate standard deviation ratio for testing data
#     std_ratio_te = np.std(sim_test) / np.std(y_test)
#     # Calculate mean ratio for testing data
#     mean_ratio_te = np.mean(sim_test) / np.mean(y_test)
#     # Calculate KGE for testing data
#     KGE_te = 1 - np.sqrt((corr_coef_te - 1) ** 2 + (std_ratio_te - 1) ** 2 + (mean_ratio_te - 1) ** 2)
    
#     scores = scores.append({'code_bss': code,
#                             'Class':Class,
#                             'NSE_tr': NSE_tr,
#                             'KGE_tr': KGE_tr,
#                             'MAE_tr': MAE_tr,
#                             'R2_tr': R2_tr,
#                             'RMSE_tr': RMSE_tr,
#                             'NRMSE_tr': nrmse_tr,
#                             'NSE_te': NSE_te.values[0],
#                             'KGE_te': KGE_te.values[0],
#                             'MAE_te': MAE_te,
#                             'R2_te': R2_te,
#                             'RMSE_te': RMSE_te,
#                             'NRMSE_te':NRMSE_te.values[0]}, ignore_index=True)
#     scores.to_csv('scores_era5'+str(modtype)+'.csv')
    
#     # scores_tr = pd.DataFrame(np.array([[ R2_tr, RMSE_tr, MAE_tr]]),
#     #                    columns=['R2_tr','RMSE_tr','MAE_tr'])
#     # scores_te = pd.DataFrame(np.array([[ R2_te, RMSE_te, MAE_te]]),
#     #                    columns=['R2_te','RMSE_te','MAE_te'])
#     # scores_te = scores_te.append(pd.DataFrame(np.array([[ R2_te, RMSE_te, MAE_te]]), columns=['R2_te','RMSE_te','MAE_te']), ignore_index=True)
#     # scores_te.to_csv('scores_te_era5'+str(modtype)+'{}.csv'.format(code.replace('/','_')))

#     from matplotlib import pyplot
#     pyplot.figure(figsize=(20,6))
#     Train_index=X_train.date
#     Test_index=X_test.date
#     # Extend_index=X_extend.date
#     from matplotlib import rc
#     # rc('font',**{'family':'serif'})
#     # rc('text', usetex = True)

  
#     pyplot.plot(df_x1['date'], df_x1['val_calc_ngf'],linestyle=':', marker='*', label=code)     
#     # pyplot.plot( Train_index,y_train, 'k', label ="observed train", linewidth=1.5,alpha=0.9)

#     pyplot.fill_between(Train_index[seq_length-1:],lower_bound_train,
#                     upper_bound_train, facecolor = (1.0, 0.8549, 0.72549),
#                     label ='95% confidence training',linewidth = 1,
#                     edgecolor = (1.0, 0.62745, 0.47843))    
#     pyplot.plot( Train_index[seq_length-1:],sim_train, 'r', label ="simulated  median train", linewidth = 0.9)
#     pyplot.plot( Test_index,sim, 'r', label ="simulated mean", linewidth = 0.9)

#     # Convert Timestamp to datetime
#     train_last_date = Train_index.iloc[-1].to_pydatetime()

#     pyplot.fill_between( Test_index,lower_bound_test,
#                     upper_bound_test, facecolor = (1,0.7,0,0.4),
#                     label ='95% confidence',linewidth = 1,
#                     edgecolor = (1,0.7,0,0.7))    
#     # pyplot.plot( Extend_index[seq_length-1:],sim_extend, 'g', label ="reconstructed mean", linewidth = 0.9)
#     bbox_props = dict(boxstyle='square', facecolor='white', alpha=0.1)
#     textbox_x = train_last_date + pd.DateOffset(months=90)  # Adjust the offset as needed
#     textbox_y = y_train.max() * 1.0  # Adjust the y-coordinate as needed
#     pyplot.text(textbox_x, textbox_y, f'KGE_te: {float(KGE_te):.2f}', fontsize=12,  color='black', weight='bold',bbox=bbox_props)
#     pyplot.text(textbox_x, textbox_y - (y_train.max() * 0.02), f'MAE_te: {MAE_te:.2f}', fontsize=12, color='black', weight='bold', bbox=bbox_props)
#     pyplot.text(textbox_x, textbox_y - (y_train.max() * 0.04), f'NRMSE_te: {float(NRMSE_te):.2f}', fontsize=12, color='black', weight='bold', bbox=bbox_props)

#     # pyplot.plot( Extend_index[seq_length-1:],sim_extend, 'g', label ="reconstructed mean", linewidth = 0.9)
    
            
    
#     # pyplot.fill_between( Extend_index[seq_length-1:],sim_extend.reshape(-1,) - y_err_ex,
#     #                 sim_extend.reshape(-1,) + y_err_ex, facecolor = (1,0.1,0,0.2),
#     #                 label ='95% confidence extend',linewidth = 1,
#     #                 edgecolor = (1,0.2,0,0.2))    

#     pyplot.vlines(x=Test_index.iloc[0], ymin=[y_train.min()*0.999], ymax=[y_train.max()*1.001], colors='teal', ls='--', lw=2, label='Start of Testing data')
#     # pyplot.vlines(x=Train_index[seq_length-1:].iloc[0], ymin=[y_train.min()*0.999], ymax=[y_train.max()*1.001], colors='red', ls='--', lw=2, label='Start of Training data')
#     pyplot.title((str(modtype)+' '+ "ERA5 OHE"+str(code)), size=24,fontweight = 'bold')
#     pyplot.xticks(fontsize=14)
#     pyplot.yticks(fontsize=14)
#     pyplot.ylabel('GWL [m asl]', size=24)
#     pyplot.xlabel('Date',size=24)
#     pyplot.legend()
#     pyplot.tight_layout()
    
#     pyplot.savefig('./'+'Well_ID_era5_OHE_static'+str(modtype)+str(GWL_type)+str(initm)+'{}.png'.format(code.replace('/','_')), dpi=600, bbox_inches = 'tight', pad_inches = 0.1)




#     # plt.colorbar()
    
    
#     plt.show()



# fig, axs = plt.subplots(nrows=1, ncols=len(final_df['code_bss'].unique()), figsize=(20,5))

# i = 0
# for code in final_df['code_bss'].unique():

#     df = final_df[final_df['code_bss'] == code]
#     Xcorr = df.drop(columns=['code_bss','qualification'])
#     # Create a correlation matrix between the variables in df2
#     corr = Xcorr.corr()

#     corr_specific_variable = corr['val_calc_ngf']
#     corr_specific_variable = corr_specific_variable.drop('val_calc_ngf')
#     corr_specific_variable.plot.bar(ax=axs[i])
#     axs[i].set_title(code, size=24, fontweight='bold')
#     i += 1

#     # Use seaborn to create a heatmap of the correlation matrix
#     # sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu')

# import matplotlib.pyplot as plt

# for code in final_df['code_bss'].unique():
#     df = final_df[final_df['code_bss'] == code]
#     # plt.plot(df['date'], df['val_calc_ngf'], label=code)
#     plt.plot(df['date'], df['sro'], label='sro'+str(code))
# plt.xlabel('Date')
# plt.ylabel('val_calc_ngf')
# plt.legend()
# plt.show()
