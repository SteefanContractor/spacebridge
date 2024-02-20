# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
col_names = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [col_names[i] for i in list(range(18))] 
lab_names = [data_lab_name[l] for l in label]
col_names = orig_feats + lab_names
dtype = dict({c: 'float64' for c in col_names if c not in ['date']} , **{c: 'float32' for c in col_names if c in data_lab_name.values()})
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=col_names,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[col_names] # reorder columns
data[lab_names] = data[lab_names]/100.
data['water_conc'] = 1. - data[lab_names].sum(axis=1)
lab_names.append('water_conc')
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = orig_feats[2:-2] # original features are the features used for training; so no labels or data/time
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
# orig_data = data.copy()
label = data[lab_names]
label = label.idxmax(axis=1)
# drop date, time, lat, lon columns
data.drop(['date','time','latitude','longitude']+lab_names, axis=1, inplace=True)
# %%
# import PCA object
with open('../products/models/sk_bgmm/pca_7comp.pkl','rb') as f:
    pca = pickle.load(f)
# %%
# transform data
data = pca.transform(data)
# %%
# import bgmm object
with open('../products/models/sk_bgmm/bgmm_2comp_200iter_PCA7comp.pkl','rb') as f:
    bgmm = pickle.load(f)
# %%
# predict probabilities
P = bgmm.predict_proba(data)
K = P.shape[1]
C = len(lab_names)
# the following R matrix taken from the result of the optimization in rmda_07Feb24.ipynb 
# this particular result is taken from trus-constr minimization which gave the highest likelihood of -5240826.027765692
R = np.array([[0.02374023, 0.25716218],
 [0.01122226, 0.58935709],
 [0.00606944, 0.04607074],
 [0.95896806, 0.10741   ],])
# %%
def rmda_predict(gmm, R, X):
    """
    Predict the classes and associated probabilities for the given data X using the given GMM and R matrix
    """
    P = gmm.predict_proba(X)
    K = P.shape[1]
    C = R.shape[0]
    Pfin = P @ R.T
    return Pfin.argmax(axis=1), Pfin
# %%
# predict classes and probabilities
rmda_label, rmda_prob = rmda_predict(bgmm, R, data)
# %%
### bayesian update original concentrations with the rmda results

# %%
# import original concentrations
data_path = '../data/'
label = ['yi','fyi','myi']
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
col_names = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [col_names[i] for i in list(range(18))] 
lab_names = [data_lab_name[l] for l in label]
col_names = orig_feats + lab_names
dtype = dict({c: 'float64' for c in col_names if c not in ['date']} , **{c: 'float32' for c in col_names if c in data_lab_name.values()})
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=col_names,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[col_names] # reorder columns
data[lab_names] = data[lab_names]/100.
data['water_conc'] = 1. - data[lab_names].sum(axis=1)
lab_names.append('water_conc')
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = orig_feats[2:-2] # original features are the features used for training; so no labels or data/time
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
# drop feature columns
data.drop(columns=orig_feats, inplace=True)
data
# %%
rmda_upd_prob = rmda_prob * data[lab_names]
# normalize
rmda_upd_prob = rmda_upd_prob.div(rmda_upd_prob.sum(axis=1), axis=0)
rmda_upd_prob
# %%
# All GNSS-R observations
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
for ax, lab in zip(axs, lab_names):
  ax.hist([rmda_upd_prob[lab]], 
          bins=40, density=True, histtype='bar',label=lab,)
  ax.set_xlabel('ice probabilities (%)')
  ax.legend()
plt.tight_layout()
# %%
# Only non-zero GNSS-R observations
tot_conc = rmda_upd_prob[lab_names[:-1]].sum(axis=1)
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
for ax, lab in zip(axs, lab_names):
  ax.hist([rmda_upd_prob.loc[tot_conc > 0][lab]], 
          bins=40, density=True, histtype='bar',label=lab,)
  ax.set_xlabel('Non zero ice probabilities (%)')
  ax.legend()
plt.tight_layout()
# %%
