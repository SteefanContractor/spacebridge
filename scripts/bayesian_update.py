# %%
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from datetime import datetime
# %%
# load preprocessed data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
lab_names =  ['YI_conc','FYI_conc','MYI_conc','water_conc']
labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype='float32', usecols=['index']+lab_names)
labels.index = labels.index.astype('int64')
labels = labels[labels.water_conc < 100.]
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype='float64')
features.index = features.index.astype('int64')
features = features.loc[labels.index]
len(features), len(labels)
# %%
display(features.head())
display(labels.head())
# %% 
with open('../products/models/rmda/rmda_umapfeats_K4C4_trust-constr_20240604:171926.pkl', 'rb') as f:
  result = pickle.load(f)
# %%
# load lgbm model
with open('../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_20240508:082525.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
# load sk_bgmm model of rmda model
with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl','rb') as f:
    bgmm_model = pickle.load(f)
# %%
lgbm_confusion_mat = np.array([[0.67241379, 0.15922921, 0.04158215, 0.12677485],
                            [0.04790419, 0.84677703, 0.08559352, 0.01972526],
                            [0.046875,   0.25878906, 0.64257812, 0.05175781],
                            [0.01741254, 0.00994283, 0.00626461, 0.96638002]])
R = np.array([[0.30267574, 0.01846785, 0.32744918, 0.15491719],
            [0.33155807, 0.00132933, 0.37493539, 0.0345478 ],
            [0.35329142, 0.00568585, 0.28904677, 0.15025457],
            [0.01247476, 0.97451697, 0.00856866, 0.66028044]])
# %%
# predict proba lgbm
# permutation according to lgbm_model.classes_ because labels are in the wrong order
permutation = [int(np.where(lgbm_model.classes_ == l)[0]) for l in lab_names]
lgbm_proba = lgbm_model.predict_proba(features)[:, permutation]
# perdict proba bgmm
bgmm_proba = bgmm_model.predict_proba(features)
# %%
# lgbm
lgbm_post = (labels * (lgbm_confusion_mat @ lgbm_proba.T).T)
lgbm_post = lgbm_post.div(lgbm_post.sum(axis=1), axis=0)
# rmda
rmda_post = (labels * (R @ bgmm_proba.T).T)
rmda_post = rmda_post.div(rmda_post.sum(axis=1), axis=0)
# %%
# save results
timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
print(f'Saving with timestamp: {timestamp}')
lgbm_proba = pd.DataFrame(lgbm_proba, index=features.index, columns=lab_names)
bgmm_proba = pd.DataFrame(bgmm_proba, index=features.index, columns=lab_names)
# lgbm_proba.to_csv(f'../products/results/lgbm_predict_proba_{timestamp}.csv')
# bgmm_proba.to_csv(f'../products/results/bgmm_predict_proba_{timestamp}.csv')
# lgbm_post.to_csv(f'../products/results/lgbm_posterior_proba_{timestamp}.csv')
# rmda_post.to_csv(f'../products/results/rmda_posterior_proba_{timestamp}.csv')
# %%
# load posterior proba
lgbm_post = pd.read_csv('../products/results/lgbm_posterior_proba_20240513:113448.csv', index_col=0)
lgbm_post = lgbm_post*100.
rmda_post = pd.read_csv('../products/results/withoutumap_rmda_posterior_proba_20240815:130426.csv', index_col=0)
rmda_post = rmda_post*100.
# %%
# load posterior proba
lgbm_post = pd.read_csv('../products/results/umap_lgbm_posterior_proba_20240607:090930.csv', index_col=0)
lgbm_post = lgbm_post*100.
rmda_post = pd.read_csv('../products/results/umap_rmda_posterior_proba_20240607:090930.csv', index_col=0)
rmda_post = rmda_post*100.
# %%
labels['tot_ice_conc'] = labels.drop(columns=['water_conc']).sum(axis=1)
lgbm_post['tot_ice_conc'] = lgbm_post.drop(columns=['water_conc']).sum(axis=1)
rmda_post['tot_ice_conc'] = rmda_post.drop(columns=['water_conc']).sum(axis=1)
# %%
# histogram of updated concentrations
ice_lab_names = ['tot_ice_conc','YI_conc','FYI_conc','MYI_conc']
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('IUP Multiage Sea Ice Concentration', verticalalignment='center')
for ax, lab in zip(axs, ice_lab_names):
  ax.hist([labels.loc[labels[lab]>0,lab],
           lgbm_post.loc[lgbm_post[lab]>0,lab],
           rmda_post.loc[rmda_post[lab]>0,lab]], 
          bins=40, density=False, histtype='bar',label=['IUP original','LGBM updated','RMDA updated'],)
  ax.set_xlabel('Non zero '+lab+' (%)')
  ax.legend() if lab=='tot_ice_conc' else None
plt.tight_layout()
# %%
print('Original U.Brem ice concentration')
display(labels.idxmax(axis=1).value_counts())
print('LGBM updated ice concentration')
display(lgbm_post.idxmax(axis=1).value_counts())
print('RMDA updated ice concentration')
display(rmda_post.idxmax(axis=1).value_counts())
# %%
print('Original U.Brem ice concentration')
display({'YI_conc': labels[labels.YI_conc > 90.].shape[0],
'FYI_conc': labels[labels.FYI_conc > 99.9].shape[0],
'MYI_conc': labels[labels.MYI_conc > 99.].shape[0],
'water_conc': labels[labels.water_conc == 100.].shape[0]})
print('LGBM updated ice concentration')
display({'YI_conc': lgbm_post[lgbm_post.YI_conc > 90.].shape[0],
'FYI_conc': lgbm_post[lgbm_post.FYI_conc > 99.9].shape[0],
'MYI_conc': lgbm_post[lgbm_post.MYI_conc > 99.].shape[0],
'water_conc': lgbm_post[lgbm_post.water_conc == 100.].shape[0]})
print('RMDA updated ice concentration')
display({'YI_conc': rmda_post[rmda_post.YI_conc > 90.].shape[0],
'FYI_conc': rmda_post[rmda_post.FYI_conc > 99.9].shape[0],
'MYI_conc': rmda_post[rmda_post.MYI_conc > 99.].shape[0],
'water_conc': rmda_post[rmda_post.water_conc == 100.].shape[0]})
# %%
print('Original U.Brem ice concentration')
display(labels.describe())
print('LGBM updated ice concentration')
display(lgbm_post.describe())
print('RMDA updated ice concentration')
display(rmda_post.describe())
# %%
# load metadata
metadata = pd.read_csv(data_path + "metadata.csv", index_col=0, parse_dates=['date'])
metadata = metadata.loc[labels.index]
winter = metadata.query('20200801 <= date <= 20201031')
summer = metadata.query('20200101 <= date <= 20200331') #date < 20200801 or date >= 20201101
shoulder = metadata.drop(pd.concat([winter, summer]).index)
assert len(winter)+len(summer)+len(shoulder) == len(metadata)
# %%
label=['tot_ice_conc','YI_conc','FYI_conc','MYI_conc']
for data, title in zip([labels, lgbm_post, rmda_post], ['U.Brem Original','LGBM Updated','RMDA Updated']):
  winterdata = data.loc[winter.index]
  summerdata = data.loc[summer.index]
  shoulderdata = data.loc[shoulder.index]
  fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
  fig.suptitle(f'{title} Ice Concentration', verticalalignment='center')
  for ax, lab in zip(axs, label):
    ax.hist([data.loc[data[lab]>0,lab], winterdata.loc[winterdata[lab]>0,lab], summerdata.loc[summerdata[lab]>0,lab], shoulderdata.loc[shoulderdata[lab]>0,lab]], 
            bins=40, density=True, histtype='bar',
            label=[f'All 2020; N={len(data)}', f'Winter (ASO); N={len(winterdata)}', f'Summer (JFM); N={len(summerdata)}', f'Shoulder (AMJJ+ND); N={len(shoulderdata)}'])
    ax.set_xlabel('Non zero '+lab+' ice concentration (%)')
    ax.legend() if lab in ['MYI_conc'] else None
  plt.tight_layout()
  plt.show()
# %%
# high confidence ice data
hci_data = {}
for d in ['orig', 'lgbm', 'rmda']:
  hci_data[d] = {}
  data = labels if d == 'orig' else lgbm_post if d == 'lgbm' else rmda_post
  hci_data[d]['all'] = data[(data.YI_conc > 90.) | (data.FYI_conc > 99.9) | (data.MYI_conc > 99.)][['YI_conc','FYI_conc','MYI_conc']].idxmax(axis=1).to_frame().rename(columns={0:'label'})
  hci_data[d]['all'][['date','longitude','latitude']] = metadata.loc[hci_data[d]['all'].index, ['date','longitude','latitude']]
  hci_data[d]['winter'] = hci_data[d]['all'].query('20200801 <= date <= 20201031')
  hci_data[d]['summer'] = hci_data[d]['all'].query('20200101 <= date <= 20200331')
  hci_data[d]['shoulder'] = hci_data[d]['all'].drop(pd.concat([hci_data[d]['winter'], hci_data[d]['summer']]).index)
# %%
seas_legend = ['All 2020', 'Winter (ASO)', 'Summer (JFM)', 'Shoulder (AMJJ+ND)']
seas = ['all', 'winter', 'summer', 'shoulder']
for d in ['orig', 'lgbm', 'rmda']:
  fig, axs = plt.subplots(4,3,subplot_kw=dict(projection=ccrs.Orthographic(180,-90)), figsize=(15,18))
  fig.subplots_adjust(wspace=0, hspace=0.05)
  for si, s in enumerate(seas):
    for li, lab in enumerate(['YI_conc','FYI_conc','MYI_conc']):
      data = hci_data[d][s].loc[hci_data[d][s].label==lab,['longitude', 'latitude']]
      # if lab == 'totm_conc':
      axs[si,li].scatter(data['longitude'], data['latitude'], transform=ccrs.PlateCarree(), c='k', alpha=0.1, s=1)
      axs[si,li].coastlines()
      if li==0:
        # axs[si,li].set_ylabel(seas_legend[si], size=100,)
        plt.annotate(seas_legend[::-1][si], (0.07, si/4+0.07), xycoords='subfigure fraction', fontsize=12, rotation=90)
        # axs[si,li].yaxis.set_label_coords(0.5,1.02)
      if si==0:
        axs[si,li].set_title(lab)
  plt.tight_layout()
  title = 'Original U.Brem data' if d == 'orig' else 'LGBM updated data' if d == 'lgbm' else 'RMDA updated data'
  fig.suptitle(title, fontsize=24, y=0.92)
  plt.show()
# %%
# IUP total ice concentration > 80% colored by majority ice class for winter and summer
labels = labels.loc[(labels.YI_conc > 80.) | (labels.FYI_conc > 80.) | (labels.MYI_conc > 80.)]
lgbm_post = lgbm_post.loc[(lgbm_post.YI_conc > 80.) | (lgbm_post.FYI_conc > 80.) | (lgbm_post.MYI_conc > 80.)]
rmda_post = rmda_post.loc[(rmda_post.YI_conc > 80.) | (rmda_post.FYI_conc > 80.) | (rmda_post.MYI_conc > 80.)]
# join with metadata to get the lat/lon
labels = labels.join(metadata[['longitude','latitude', 'date']])
lgbm_post = lgbm_post.join(metadata[['longitude','latitude', 'date']])
rmda_post = rmda_post.join(metadata[['longitude','latitude', 'date']])
# create majority column
labels['majority'] = labels[['YI_conc','FYI_conc','MYI_conc','water_conc']].idxmax(axis=1)
lgbm_post['majority'] = lgbm_post[['YI_conc','FYI_conc','MYI_conc','water_conc']].idxmax(axis=1)
rmda_post['majority'] = rmda_post[['YI_conc','FYI_conc','MYI_conc','water_conc']].idxmax(axis=1)
# %%
def plot_pack_ice(data, title):
  fig, axs = plt.subplots(1,2,subplot_kw=dict(projection=ccrs.Orthographic(0,-90)), figsize=(15,7))
  for i, seas in enumerate(['winter', 'summer']):
    plot_data = data.query('20200801 <= date <= 20201031') if seas == 'winter' else data.query('20200101 <= date <= 20200331')
    scatter = axs[i].scatter(x=plot_data['longitude'], y=plot_data['latitude'], 
                   c=plot_data['majority'].astype('category').cat.codes,
                   transform=ccrs.PlateCarree(), alpha=0.01, s=1)
    handles, _ = scatter.legend_elements(prop="colors", alpha=1.0)
    axs[i].set_title(seas + ' (ASO)' if seas == 'winter' else seas + ' (JFM)')
    axs[i].coastlines()
    axs[i].set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
    axs[i].legend(handles, plot_data['majority'].astype('category').cat.categories, loc='upper right') if seas=='summer' else None
  fig.suptitle(f'Only locations with at least 80% of one type of {title} ice conc', fontsize=20)
# %%
plot_pack_ice(labels, 'IUP')
# %%
plot_pack_ice(lgbm_post, 'LGBM updated')
# %%
plot_pack_ice(rmda_post, 'RMDA updated')
# %%
labels.majority.value_counts()
# %%
lgbm_post.majority.value_counts()
# %%
rmda_post.majority.value_counts()
# %%

# %%
# # load preprocessed data
# data_path = '../data/'
# label = ['yi','myi','fyi']
# data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
#                      yi='YI_conc',
#                      myi='MYI_conc',
#                      fyi='FYI_conc')
# # read the header to get the column names
# orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
# orig_feats = [orig_feats[i] for i in list(range(18))] + [data_lab_name[l] for l in label]
# dtype = dict({c: 'float64' for c in orig_feats if c not in list(data_lab_name.values())+['date']}, **{c: 'float32' for c in orig_feats if c in data_lab_name.values()})
# # # read values
# data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
# data = data[orig_feats] # reorder columns
# # drop -999.0 values from reflectivity1 and reflectivity2 columns
# data = data[data.reflectivity1!=-999.0]
# data = data[data.reflectivity2!=-999.0]
# data.dropna(inplace=True)
# orig_feats = data.keys()[2:-5] # original features are the features used for training; so no labels or data/time
# labels = [data_lab_name[l] for l in label]
# if len(labels)>1:
#     data['tot_conc'] = data[labels].sum(axis=1)
# data.reset_index(inplace= True, drop=True)
# data
# # %%
# # log transform snr_reflected and power_reflected columns
# data['snr_reflected1'] = np.log10(data['snr_reflected1'])
# data['snr_reflected2'] = np.log10(data['snr_reflected2'])
# data['power_reflected1'] = np.log10(data['power_reflected1'])
# data['power_reflected2'] = np.log10(data['power_reflected2'])
# data['reflectivity1'] = np.log10(data['reflectivity1'])
# data['reflectivity2'] = np.log10(data['reflectivity2'])
# # drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
# data.dropna(inplace=True)# %%
# # %%
# # drop rows with excess_phase_noise2 < -5 (only 35 rows)
# data = data[data.excess_phase_noise2>-5.]
# # %%
# # min-max scale original features
# scaler = MinMaxScaler()
# data[orig_feats] = scaler.fit_transform(data[orig_feats])
# %%
# subset data to only include non zero ice
# data = data[data.tot_conc>0.]
# %%
# load waterice model
# with open('../products/models/train_gradboost/lgbm_clf_waterice_label_2023-09-27.pkl', 'rb') as f:
#     waterice_model = pickle.load(f)
# # load updated distinct ice type model
# with open('../products/models/train_gradboost/lgbm_clf_updated_icetype_2023-10-10.pkl', 'rb') as f:
#     ice_model = pickle.load(f)
    
# # %%
# # get model predictions on unsure waterice data
# data['waterice_pred'] = waterice_model.predict(data[orig_feats], axis=1)
# # get model predictions on ice data
# data['ice_pred'] = ice_model.predict(data[orig_feats], axis=1)
# # %%
# waterice_val_confusion_matrix = np.array([[0.97546399, 0.08976091],
#                                  [0.02453601, 0.91023909]])
# # use the val prediction normalized confusion matrix to update the ice concentrations
# ice_val_confusion_matrix = np.array([[0.71908652, 0.22368421, 0.12334495],
#                                  [0.18455254, 0.65935673, 0.12404181],
#                                  [0.09636094, 0.11695906, 0.75261324]])
# # %%
# # update the ice concentrations using the val_confusion_matrix
# data['updated_conc'] = 100.*waterice_val_confusion_matrix[1,data.waterice_pred]*data.tot_conc/(waterice_val_confusion_matrix[0,data.waterice_pred]*(100.-data.tot_conc)+waterice_val_confusion_matrix[1,data.waterice_pred]*data.tot_conc)
# # scale up the ice type concentrations
# data['updated_YI'] = data['YI_conc']/data['tot_conc']*data['updated_conc']
# data['updated_FYI'] = data['FYI_conc']/data['tot_conc']*data['updated_conc']
# data['updated_MYI'] = data['MYI_conc']/data['tot_conc']*data['updated_conc']
# # %%
# # update the ice concentrations using the val_confusion_matrix
# # data['updated_conc'] = 100.*val_confusion_matrix[1,data.waterice_pred]*data.tot_conc/(val_confusion_matrix[0,data.waterice_pred]*(100.-data.tot_conc)+val_confusion_matrix[1,data.waterice_pred]*data.tot_conc)
# # scale up the ice type concentrations
# data['ice_upd_FYI'] = data['updated_FYI']*ice_val_confusion_matrix[0,data.ice_pred]
# data['ice_upd_MYI'] = data['updated_MYI']*ice_val_confusion_matrix[1,data.ice_pred]
# data['ice_upd_YI'] = data['updated_YI']*ice_val_confusion_matrix[2,data.ice_pred]
# normalizer = np.sum(data[['ice_upd_FYI','ice_upd_MYI','ice_upd_YI']],axis=1)
# data['ice_upd_FYI'] = data['ice_upd_FYI']/normalizer*data['updated_conc']
# data['ice_upd_MYI'] = data['ice_upd_MYI']/normalizer*data['updated_conc']
# data['ice_upd_YI'] = data['ice_upd_YI']/normalizer*data['updated_conc']
# data['ice_upd_conc'] = data['ice_upd_FYI']+data['ice_upd_MYI']+data['ice_upd_YI']
# %%
# histogram of updated concentrations
# orig=['tot_conc','YI_conc','MYI_conc','FYI_conc']
# watericeupd=['updated_conc','updated_YI','updated_MYI','updated_FYI']
# iceupd=['ice_upd_conc','ice_upd_FYI','ice_upd_MYI','ice_upd_YI']
# fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
# fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
# for ax, ilab in zip(axs, range(len(orig))):
#   ax.hist([data.loc[data[orig[ilab]]>0,orig[ilab]],
#            data.loc[data[watericeupd[ilab]]>0,watericeupd[ilab]],
#            data.loc[data[iceupd[ilab]]>0,iceupd[ilab]]], 
#           bins=40, density=True, histtype='bar',label=['original','waterice_upd','icetype_upd'],)
#   ax.set_xlabel('Non zero '+orig[ilab]+' ice concentration (%)')
#   ax.legend() if ilab==0 else None
# plt.tight_layout()