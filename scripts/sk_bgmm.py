# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import pickle
import time
import datetime
# %%
data_path = "../data/preprocessed_gnssr_update202330_clean/"
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
features = Xres.sample(frac=0.65, random_state=42)
len(features)
# %%
# load preprocessed data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
dtype = {label: 'float32' for label in lab_names}
labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype=dtype)
# %%
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "pca_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "pca_feats.csv", index_col=0, dtype=dtype)
# %%
# bic score
def gmm_bic_score(estimator, X):
    return estimator.score(X) - 0.5 * estimator.n_components * np.log(X.shape[0])
# %%
# base model
bgm = BayesianGaussianMixture(max_iter=1000, random_state=42)
# set up grid search parameters
param_grid = {'n_components': range(1,7)}
# grid search
grid_search = GridSearchCV(
    bgm, param_grid=param_grid, scoring=gmm_bic_score
)
start = time.time()
grid_search.fit(features.to_numpy())
end = time.time()
print(f'Grid search took {end-start} seconds')
# %%
timestamp = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
print(timestamp)
# save model
pickle.dump(grid_search, open(f'../products/models/sk_bgmm/gridsearch_bgm_clean_alldata_{timestamp}.pkl', 'wb'))
# %%
# base model
bgm = BayesianGaussianMixture(max_iter=500, random_state=42)
# set up grid search parameters
param_grid = {'n_components': range(1,7)}
# grid search
grid_search = GridSearchCV(
    bgm, param_grid=param_grid, scoring=gmm_bic_score
)
start = time.time()
grid_search.fit(features[labels.ice_conc > 0].to_numpy())
end = time.time()
print(f'Grid search took {end-start} seconds')
# %%
timestamp = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
print(timestamp)
# save model
pickle.dump(grid_search, open(f'../products/models/sk_bgmm/gridsearch_bgm_clean_nonzeroice_{timestamp}.pkl', 'wb'))
# %%
# load model
grid_search2 = pickle.load(open(f'../products/models/sk_bgmm/gridsearch_bgm_clean_alldata_{timestamp}.pkl', 'rb'))

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
