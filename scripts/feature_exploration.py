#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# %%
# load preprocessed data
data_path = 'data/'
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
# drop rows with excess_phase_noise2 < -5 (only 35 rows)
data = data[data.excess_phase_noise2>-5.]
# %%
# log transform snr_reflected and power_reflected columns
data['snr_reflected1'] = np.log10(data['snr_reflected1'])
data['snr_reflected2'] = np.log10(data['snr_reflected2'])
data['power_reflected1'] = np.log10(data['power_reflected1'])
data['power_reflected2'] = np.log10(data['power_reflected2'])
data['reflectivity1'] = np.log10(data['reflectivity1'])
data['reflectivity2'] = np.log10(data['reflectivity2'])
# drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
data.dropna(inplace=True)
# %%
# min-max scale original features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[orig_feats] = scaler.fit_transform(data[orig_feats])
# %%
# plot histograms of feature columns
data[orig_feats].hist(figsize=(20,20),bins=50)
# %%
# plot heatmap of feature correlations
plt.figure(figsize=(20,20))
sns.heatmap(data[orig_feats].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# %%
# heatmap of feature correlations with labels
plt.figure(figsize=(20,20))
sns.heatmap(data[list(orig_feats)+labels].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# %%
# filter tot_conc == 0 indicating water and plot histogram of original features
data[data.tot_conc==0.][orig_feats].hist(figsize=(20,20),bins=50)
# %%
# filter tot_conc > 99. indicating ice and plot histogram of original features
data[data.tot_conc>99.][orig_feats].hist(figsize=(20,20),bins=50)
# %%
# subset data to only include water and ice with equal fractions of each
waterice = data[(data.tot_conc==0.) | (data.tot_conc>99.)]
waterice['waterice_label'] = [0 if tc==0. else 1 for tc in waterice.tot_conc]
waterice = waterice.groupby('waterice_label').sample(frac=0.2,random_state=42)
# %%
# project data rows onto 2D space using tsne
# tsne features
tsne_feats = ['reflectivity1','snr_reflected1','power_reflected1','phase_noise1','excess_phase_noise1',]
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300)
tsne_results = tsne.fit_transform(waterice[tsne_feats])
# %%
# plot tsne results
plt.figure(figsize=(10,10))
plt.scatter(tsne_results[:,0],tsne_results[:,1],c=waterice['waterice_label'])
# %%
# project data rows onto 2D space using PCA
# PCA features
pca_feats = ['reflectivity1','snr_reflected1','power_reflected1','phase_noise1','excess_phase_noise1',]
pca = PCA(n_components=2)
pca_results = pca.fit_transform(waterice[pca_feats])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# %%
# plot PCA results
plt.figure(figsize=(10,10))
plt.scatter(pca_results[:,0],pca_results[:,1],c=waterice['waterice_label'])
# %%
