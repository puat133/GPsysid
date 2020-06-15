import numpy as np
import pandas as pd
import GPutils as utils
import Dirichlet

temperature_columns = np.array(
    [['TI8585','TI8553','TI8554','TI8555','TI8556','TI8557','TI8558','TI8559', 'TIZ8578A'],
     ['TI8585','TI8560','TI8561','TI8562','TI8563','TI8564','TI8565','TI8566', 'TIZ8578A'],
     ['TI8585','TI8567','TI8568','TI8569','TI8570','TI8571','TI8572','TI8573', 'TIZ8578A']],dtype=object)

tc_heights =np.array([[7600,6550,5500,4450,3400,2350,1300],
            [7250,6250,5150,4100,3050,2000,950],
            [6900,5850,4800,3750,2700,1650,600]])

tc_total_height=8000 #assumption
normalized_height = (tc_total_height-tc_heights)/tc_total_height
Delta_z = normalized_height - np.hstack((np.zeros((3,1)),normalized_height[:,0:-1]))

training_pole_temperature_positions = normalized_height[0,:]
validation_pole_temperature_positions = normalized_height[1,:]

all_temperature_positions = normalized_height.flatten()


L = 1.

#Load data
df_raw = pd.read_hdf('Data/timeseries_complete.hdf5',key='KAAPO_hour_15_16_17_18_19_complete')
df_raw = df_raw[(df_raw.index < "2017-03-26") & (df_raw.index > "2015-07-14")]
df_lab = pd.read_hdf('Data/Laboratory.hdf5',key='Laboratory').interpolate()
df_lab = df_lab[(df_lab.index < "2017-03-26") & (df_lab.index > "2015-07-14")]
df = pd.concat([df_raw, df_lab], axis=1)
df = df.resample('d').median() #resample daily or weekly

y_hist_first = df[temperature_columns[0,1:-1].flatten()].values
y_hist_first_ma = utils.moving_average(y_hist_first,window=21)

y_hist_second = df[temperature_columns[1,1:-1].flatten()].values
y_hist_second_ma = utils.moving_average(y_hist_second,window=21)

y_hist_thrid = df[temperature_columns[2,1:-1].flatten()].values
y_hist_thrid_ma = utils.moving_average(y_hist_thrid,window=21)

threshold = 1e-1
input_hist = df[['sulphur feed max','AROM-LC wt-%','TI8585']].fillna(0).values + threshold
temp_inlet = df['TI8585'].values
temp_outlet = df['TIZ8578A'].values
# temp_reference = temp_inlet + training_pole_temperature_positions[:,np.newaxis]*(temp_outlet-temp_inlet)
temp_reference =  Dirichlet.construct_reference_temperature(temp_inlet,temp_outlet,training_pole_temperature_positions)

y_hist_normalized = y_hist_first.T - temp_inlet
y_hist_dirichlet = y_hist_first.T - temp_reference

y_hist = df[temperature_columns[:,1:-1].flatten()].values
all_y_hist_normalized = y_hist.T - temp_inlet