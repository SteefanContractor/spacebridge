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
# subset data to only include water and ice with equal fractions of each
sure_waterice = data[(data.tot_conc==0.) | (data.tot_conc>99.)]
unsure_waterice = data[(data.tot_conc>0.) & (data.tot_conc<=99.)]
# %%
# load waterice model
with open('../products/models/train_gradboost/lgbm_clf_waterice_label_2023-09-27.pkl', 'rb') as f:
    waterice_model = pickle.load(f)
# %%
# drop the first two columns and the last six columns except the very last column
unsure_waterice.drop(data.columns[:2].tolist()+data.columns[-6:-4].tolist(), axis=1,inplace=True)
sure_waterice.drop(data.columns[:2].tolist()+data.columns[-6:-4].tolist(), axis=1,inplace=True)
# %%
# get model predictions on unsure waterice data
data['waterice_pred'] = waterice_model.predict(data.drop(data.columns[:2].tolist()+data.columns[-6:].tolist(), axis=1))
# %%
# use the val prediction normalized confusion matrix to update the ice concentrations
val_confusion_matrix = np.array([[0.97546399, 0.08976091],
                                 [0.02453601, 0.91023909]])
# %%
# update the ice concentrations using the val_confusion_matrix
data['updated_conc'] = 100.*val_confusion_matrix[1,data.waterice_pred]*data.tot_conc/(val_confusion_matrix[0,data.waterice_pred]*(100.-data.tot_conc)+val_confusion_matrix[1,data.waterice_pred]*data.tot_conc)
# scale up the ice type concentrations
data['updated_YI'] = data['YI_conc']/data['tot_conc']*data['updated_conc']
data['updated_FYI'] = data['FYI_conc']/data['tot_conc']*data['updated_conc']
data['updated_MYI'] = data['MYI_conc']/data['tot_conc']*data['updated_conc']
# %%
# histogram of updated concentrations
label=['updated_conc','updated_YI','updated_MYI','updated_FYI']
oldlabel=['tot_conc','YI_conc','MYI_conc','FYI_conc']
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
for ax, ilab in zip(axs, range(len(label))):
  ax.hist([data.loc[data[label[ilab]]>0,label[ilab]],data.loc[data[oldlabel[ilab]]>0,oldlabel[ilab]]], 
          bins=40, density=True, histtype='bar',label=['updated','original'],)
  ax.set_xlabel('Non zero '+oldlabel[ilab]+' ice concentration (%)')
  ax.legend() if ilab==0 else None
plt.tight_layout
# %%
# create venn diagrams to visualize the updated ice concentrations
# total concentration
venn2(subsets=(sum((data.updated_conc>99.) & (data.tot_conc<=99.)), sum((data.updated_conc>99.) & (data.tot_conc>99.)), sum((data.updated_conc<=99.) & (data.tot_conc>99.))), set_labels=('updated_conc 99% threshold','tot_conc 99% threshold'))
plt.show()
# YI
venn2(subsets=(sum((data.updated_YI>98.) & (data.YI_conc<=98.)), sum((data.updated_YI>98.) & (data.YI_conc>90.)), sum((data.updated_YI<=98.) & (data.YI_conc>90.))), set_labels=('updated_YI 98% threshold','YI_conc 90% threshold'))
plt.show()
# MYI 
venn2(subsets=(sum((data.updated_MYI>99.) & (data.MYI_conc<=99.)), sum((data.updated_MYI>99.) & (data.MYI_conc>99.)), sum((data.updated_MYI<=99.) & (data.MYI_conc>99.))), set_labels=('updated_MYI 99% threshold','MYI_conc'))
plt.show()
# FYI
venn2(subsets=(sum((data.updated_FYI>99.9) & (data.FYI_conc<=99.9)), sum((data.updated_FYI>99.9) & (data.FYI_conc>99.9)), sum((data.updated_FYI<=99.9) & (data.FYI_conc>99.9))), set_labels=('updated_FYI 99.9% threshold','FYI_conc 99.9% threshold'))
plt.show()
venn2(subsets=(sum((data.updated_FYI>99.99) & (data.FYI_conc<=99.9)), sum((data.updated_FYI>99.9) & (data.FYI_conc>99.9)), sum((data.updated_FYI<=99.99) & (data.FYI_conc>99.9))), set_labels=('updated_FYI 99.99% threshold','FYI_conc 99.9% threshold'))
plt.show()
# %%
# investigate what percentiles YI>98., MYI>99., and FYI>99.9 refer to
print(f'YI_conc > 98. results in {len(data[data.updated_YI>98.])} rows')
print(f'which corresponds to top {len(data[data.updated_YI>98.])/sum(data.updated_YI>0.)*100:.2f}% of YI_conc values in distinct_ice_types')
print(f'MYI_conc > 99. results in {len(data[data.updated_MYI>99.])} rows')
print(f'which corresponds to top {len(data[data.updated_MYI>99.])/sum(data.updated_MYI>0.)*100:.2f}% of MYI_conc values in distinct_ice_types')
print(f'FYI_conc > 99.9 results in {len(data[data.updated_FYI>99.9])} rows')
print(f'which corresponds to top {len(data[data.updated_FYI>99.9])/sum(data.updated_FYI>0.)*100:.2f}% of FYI_conc values in distinct_ice_types')
print(f'FYI_conc > 99.99 results in {len(data[data.updated_FYI>99.99])} rows')
print(f'which corresponds to top {len(data[data.updated_FYI>99.99])/sum(data.updated_FYI>0.)*100:.2f}% of FYI_conc values in distinct_ice_types')
# %%
distinct_ice_types = data[(data.updated_YI>98.) | (data.updated_MYI>99.) | (data.updated_FYI>99.99)]
distinct_ice_types['ice_type'] = ['YI' if yi>98. else 'MYI' if myi>99. else 'FYI' for yi,myi in zip(distinct_ice_types.updated_YI,distinct_ice_types.updated_MYI)]
# save distinct_ice_types dataframe to file
distinct_ice_types.to_csv('../data/updated_distinct_ice_types.csv',index=False)