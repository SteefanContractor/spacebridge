# %%
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
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
# subset data to only include non zero ice
data = data[data.tot_conc>0.]
# %%
# load waterice model
with open('../products/models/train_gradboost/lgbm_clf_waterice_label_2023-09-27.pkl', 'rb') as f:
    waterice_model = pickle.load(f)
# load updated distinct ice type model
with open('../products/models/train_gradboost/lgbm_clf_updated_icetype_2023-10-10.pkl', 'rb') as f:
    ice_model = pickle.load(f)
    
# %%
# get model predictions on unsure waterice data
data['waterice_pred'] = waterice_model.predict(data[orig_feats], axis=1)
# get model predictions on ice data
data['ice_pred'] = ice_model.predict(data[orig_feats], axis=1)
# %%
waterice_val_confusion_matrix = np.array([[0.97546399, 0.08976091],
                                 [0.02453601, 0.91023909]])
# use the val prediction normalized confusion matrix to update the ice concentrations
ice_val_confusion_matrix = np.array([[0.71908652, 0.22368421, 0.12334495],
                                 [0.18455254, 0.65935673, 0.12404181],
                                 [0.09636094, 0.11695906, 0.75261324]])
# %%
# update the ice concentrations using the val_confusion_matrix
data['updated_conc'] = 100.*waterice_val_confusion_matrix[1,data.waterice_pred]*data.tot_conc/(waterice_val_confusion_matrix[0,data.waterice_pred]*(100.-data.tot_conc)+waterice_val_confusion_matrix[1,data.waterice_pred]*data.tot_conc)
# scale up the ice type concentrations
data['updated_YI'] = data['YI_conc']/data['tot_conc']*data['updated_conc']
data['updated_FYI'] = data['FYI_conc']/data['tot_conc']*data['updated_conc']
data['updated_MYI'] = data['MYI_conc']/data['tot_conc']*data['updated_conc']
# %%
# update the ice concentrations using the val_confusion_matrix
# data['updated_conc'] = 100.*val_confusion_matrix[1,data.waterice_pred]*data.tot_conc/(val_confusion_matrix[0,data.waterice_pred]*(100.-data.tot_conc)+val_confusion_matrix[1,data.waterice_pred]*data.tot_conc)
# scale up the ice type concentrations
data['ice_upd_FYI'] = data['updated_FYI']*ice_val_confusion_matrix[0,data.ice_pred]
data['ice_upd_MYI'] = data['updated_MYI']*ice_val_confusion_matrix[1,data.ice_pred]
data['ice_upd_YI'] = data['updated_YI']*ice_val_confusion_matrix[2,data.ice_pred]
normalizer = np.sum(data[['ice_upd_FYI','ice_upd_MYI','ice_upd_YI']],axis=1)
data['ice_upd_FYI'] = data['ice_upd_FYI']/normalizer*data['updated_conc']
data['ice_upd_MYI'] = data['ice_upd_MYI']/normalizer*data['updated_conc']
data['ice_upd_YI'] = data['ice_upd_YI']/normalizer*data['updated_conc']
data['ice_upd_conc'] = data['ice_upd_FYI']+data['ice_upd_MYI']+data['ice_upd_YI']
# %%
# histogram of updated concentrations
orig=['tot_conc','YI_conc','MYI_conc','FYI_conc']
watericeupd=['updated_conc','updated_YI','updated_MYI','updated_FYI']
iceupd=['ice_upd_conc','ice_upd_FYI','ice_upd_MYI','ice_upd_YI']
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
for ax, ilab in zip(axs, range(len(orig))):
  ax.hist([data.loc[data[orig[ilab]]>0,orig[ilab]],
           data.loc[data[watericeupd[ilab]]>0,watericeupd[ilab]],
           data.loc[data[iceupd[ilab]]>0,iceupd[ilab]]], 
          bins=40, density=True, histtype='bar',label=['original','waterice_upd','icetype_upd'],)
  ax.set_xlabel('Non zero '+orig[ilab]+' ice concentration (%)')
  ax.legend() if ilab==0 else None
plt.tight_layout()