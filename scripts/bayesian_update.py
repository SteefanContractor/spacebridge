# %%
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# %%
# load preprocessed data
data_path = '../data/'
label = ['yi','myi','fyi']
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))] + [data_lab_name[l] for l in label]
dtype = dict({c: 'float64' for c in orig_feats if c not in list(data_lab_name.values())+['date']}, **{c: 'float32' for c in orig_feats if c in data_lab_name.values()})
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[orig_feats] # reorder columns
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = data.keys()[2:-5] # original features are the features used for training; so no labels or data/time
labels = [data_lab_name[l] for l in label]
if len(labels)>1:
    data['tot_conc'] = data[labels].sum(axis=1)
data.reset_index(inplace= True, drop=True)
data
# %%
# log transform snr_reflected and power_reflected columns
data['snr_reflected1'] = np.log10(data['snr_reflected1'])
data['snr_reflected2'] = np.log10(data['snr_reflected2'])
data['power_reflected1'] = np.log10(data['power_reflected1'])
data['power_reflected2'] = np.log10(data['power_reflected2'])
data['reflectivity1'] = np.log10(data['reflectivity1'])
data['reflectivity2'] = np.log10(data['reflectivity2'])
# drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
data.dropna(inplace=True)# %%
# %%
# drop rows with excess_phase_noise2 < -5 (only 35 rows)
data = data[data.excess_phase_noise2>-5.]
# %%
# min-max scale original features
scaler = MinMaxScaler()
data[orig_feats] = scaler.fit_transform(data[orig_feats])
# %%
# subset data to only include water and ice with equal fractions of each
sure_waterice = data[(data.tot_conc==0.) | (data.tot_conc>99.)]
unsure_waterice = data[(data.tot_conc>0.) & (data.tot_conc<=99.)]
# %%
# load waterice model
with open('../products/models/train_gradboost/lgbm_clf_waterice_label_2023-09-27.pkl', 'rb') as f:
    waterice_model = pickle.load(f)
# %%
# drop the first two columns and the last six columns except the very last column
unsure_waterice.drop(data.columns[:2].tolist()+data.columns[-6:-1].tolist(), axis=1,inplace=True)
sure_waterice.drop(data.columns[:2].tolist()+data.columns[-6:-1].tolist(), axis=1,inplace=True)
# %%
# get model predictions on unsure waterice data
unsure_waterice['waterice_pred'] = waterice_model.predict(unsure_waterice.drop('tot_conc', axis=1))
# %%
# use the val prediction normalized confusion matrix to update the ice concentrations
val_confusion_matrix = np.array([[0.97546399, 0.08976091],
                                 [0.02453601, 0.91023909]])
# %%
unsure_waterice['updated_conc'] = unsure_waterice.apply(lambda x: val_confusion_matrix[1,int(x['waterice_pred'])]*x['tot_conc']/
                                                        (val_confusion_matrix[1,int(x['waterice_pred'])]*x['tot_conc']+
                                                         val_confusion_matrix[0,int(x['waterice_pred'])]*(100. - x['tot_conc'])), axis=1)
unsure_waterice['updated_conc'] = unsure_waterice['updated_conc']*100.
# %%
