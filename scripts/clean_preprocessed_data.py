# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
# %%
# load preprocessed data
data_path = '../data/'
label = ['yi','fyi','myi']
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))] + [data_lab_name[l] for l in label]
dtype = {c: 'float64' for c in orig_feats if c not in ['date']}
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'])
orig_feats = data.keys()[2:-5] # original features are the features used for training; so no labels or data/time
lab_names = [data_lab_name[l] for l in label]
# %%
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
# log transform snr_reflected and power_reflected columns
data['log_snr_reflected1'] = np.log10(data['snr_reflected1'])
data['log_snr_reflected2'] = np.log10(data['snr_reflected2'])
data['log_power_reflected1'] = np.log10(data['power_reflected1'])
data['log_power_reflected2'] = np.log10(data['power_reflected2'])
data['log_reflectivity1'] = np.log10(data['reflectivity1'])
data['log_reflectivity2'] = np.log10(data['reflectivity2'])
# drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
data.dropna(inplace=True)
# drop rows with excess_phase_noise2 < -5 (only 35 rows)
data = data[data.excess_phase_noise2>-5.]
# %%
transformed_scaled_feats_list = list(orig_feats)
transformed_scaled_feats_list = ['log_'+f if f.count('reflect') > 0 else f for f in transformed_scaled_feats_list]
# min-max scale original features
scaler = MinMaxScaler()
transformed_scaled_feats = scaler.fit_transform(data[transformed_scaled_feats_list])
transformed_scaled_feats = pd.DataFrame(transformed_scaled_feats, columns=transformed_scaled_feats_list, index=data.index)
# %%
if len(lab_names)>1:
    labels = data[lab_names]
    ice_conc = labels.sum(axis=1)
    labels.loc[ice_conc > 100.] = labels[ice_conc > 100.].div(ice_conc[ice_conc > 100.],axis=0).multiply(100.,axis=0)
    labels['ice_conc'] = labels.sum(axis=1).round(13)
    labels['water_conc'] = 100. - labels['ice_conc']
# %%
# PCA transform original features
# import PCA object
with open('../products/models/sk_bgmm/pca_7comp.pkl','rb') as f:
    pca = pickle.load(f)
# transform data
pca_feats = pd.DataFrame(pca.transform(transformed_scaled_feats), columns=['pca'+str(i) for i in range(7)], index=data.index)
# %%
# metadata df
metadata = data[['date','time','latitude','longitude']]
# %%
display(data[orig_feats])
display(transformed_scaled_feats)
display(pca_feats)
display(metadata)
display(labels)
# %%
display(data[orig_feats].describe())
display(transformed_scaled_feats.describe())
display(pca_feats.describe())
display(metadata.describe())
display(labels.describe())
# %%
# write data to csv
data_path = '../data/preprocessed_gnssr_update202330_clean/'
# orig_feats
data[orig_feats].to_csv(data_path+'orig_feats.csv',index=True,index_label='index')
# transformed_scaled_feats
transformed_scaled_feats.to_csv(data_path+'transformed_scaled_feats.csv',index=True,index_label='index')
# pca_feats
pca_feats.to_csv(data_path+'pca_feats.csv',index=True,index_label='index')
# metadata
metadata.to_csv(data_path+'metadata.csv',index=True,index_label='index')
# labels
labels.to_csv(data_path+'labels.csv',index=True,index_label='index')