# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pickle
import time
# %%
# load preprocessed data
data_path = '../data/'
label = ['yi','myi','fyi']
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
# fit PCA with 3 components to 20% of data to check the total variance explained
for n_components in range(3,8):
    pca = PCA(n_components=n_components)
    pca.fit(data.sample(frac=0.2))
    # calculate total variance explained
    print(f'Total variance explained by {n_components} components: {np.sum(pca.explained_variance_ratio_)}')
# %%
for n_components in range(8,10):
    pca = PCA(n_components=n_components)
    pca.fit(data.sample(frac=0.2))
    # calculate total variance explained
    print(f'Total variance explained by {n_components} components: {np.sum(pca.explained_variance_ratio_)}')
# %%
# fit PCA with 7 components (99.3% of variance explained) and transform data
# time the PCA
start = time.time()
pca = PCA(n_components=7)
pca.fit(data)
data = pca.transform(data)
end = time.time()   
print(f'PCA took {end-start} seconds')
data = pd.DataFrame(data)

# %%
# bic score
def gmm_bic_score(estimator, X):
    return estimator.score(X) - 0.5 * estimator.n_components * np.log(X.shape[0])
# %%
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
# %%
# load model
grid_search = pickle.load(open('../products/models/sk_bgmm/gridsearch_bgm.pkl', 'rb'))

# %%
# Print the best number of components
print(grid_search.best_params_)
# %%
df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()
# %%
import seaborn as sns

sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
)
# %%
# Fit bgm with two components
start = time.time() 
bgm = BayesianGaussianMixture(n_components=4, max_iter=200, random_state=42)
bgm.fit(data)
end = time.time()   
print(f'BGMM with 4 components on data with 7 PCA components took {end-start} seconds')
# save model
pickle.dump(bgm, open('../products/models/sk_bgmm/bgmm_4comp_200iter_PCA7comp.pkl', 'wb'))
# %%
# calculate BIC score
print(f'BIC score: {gmm_bic_score(bgm, data)}')
print('converged' if bgm.converged_ else 'not converged')
# %%
# increase max_iter iteratively by 100 until converged
data = pd.DataFrame(data)
max_iter = bgm.max_iter
while not bgm.converged_:
    bgm.max_iter += 100
    start = time.time()
    bgm.fit(data.sample(frac=0.2))
    end = time.time()   
    print(f'BGMM with {max_iter} iterations took {end-start} seconds')
    print(f'BIC score: {gmm_bic_score(bgm, data)}')
    print('converged' if bgm.converged_ else 'not converged')
# %%
# fit to all data
start = time.time()
bgm.fit(data)
end = time.time()
print(f'BGMM with {max_iter} iterations took {end-start} seconds')
print(f'BIC score: {gmm_bic_score(bgm, data)}')
print('converged' if bgm.converged_ else 'not converged')
# %%
# save model
if bgm.converged_:
    pickle.dump(bgm, open('../products/models/sk_bgmm/bgmm_4comp_300iter_PCA7comp.pkl', 'wb'))
else:
    print('Model did not converge. Not saving model.')
# %%
# load model
bgm = pickle.load(open('../products/models/sk_bgmm/bgmm_4comp_300iter_PCA7comp.pkl', 'rb'))
# %%
# predict probabilities
probs = bgm.predict_proba(data)
pred_labels = bgm.predict(data)
# %%
# import labels for comparison
data_path = '../data/'
label = ['yi','myi','fyi']
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [data_lab_name[l] for l in label]
# # read values
labels = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype='float32',low_memory=True)
labels = labels/100.
labels['water_conc'] = 1. - labels.sum(axis=1)
# %%
# crosstab labels and pred_labels
pd.crosstab(label, pred_labels, normalize='columns')

# %%
