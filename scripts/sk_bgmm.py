# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
import pickle
import time
# %%
# load preprocessed data
data_path = '../data/'
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))] 
dtype = {c: 'float64' for c in orig_feats if c not in ['date']}
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[orig_feats] # reorder columns
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = data.keys()[2:-2] # original features are the features used for training; so no labels or data/time
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
# drop date, time, lat, lon columns
data.drop(['date','time','latitude','longitude'], axis=1, inplace=True)
# %%
# bic score
def gmm_bic_score(estimator, X):
    return estimator.score(X) - 0.5 * estimator.n_components * np.log(X.shape[0])
# base model
bgm = BayesianGaussianMixture(max_iter=200, random_state=42)
# set up grid search parameters
param_grid = {'n_components': range(1,7)}
# grid search
grid_search = GridSearchCV(
    bgm, param_grid=param_grid, scoring=gmm_bic_score
)
start = time.time()
grid_search.fit(data)
end = time.time()
print(f'Grid search took {end-start} seconds')
# %%
# save model
pickle.dump(grid_search, open('../products/models/sk_bgmm/gridsearch_bgm.pkl', 'wb'))