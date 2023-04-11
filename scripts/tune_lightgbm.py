# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# lightGBM seems like the best candidate as it gets the highest scores while being relatively quick
# This script tunes lightGBM models with various hyperparameters in the pycaret setting
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None


# %%
# your code here...

# %%
# load packages
from pycaret.classification import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef as mcc

# %%
import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# %%
logger.info(f'Smoke testing: {test}, smoke_test data fraction: {test_frac}')

# %%
# load preprocessed data
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv('/volstore/spacebridge/gnssr_grzice/preprocessed_data/preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(16))+[orig_feats.index(data_lab_name[label])]]
# read values
data = pd.read_csv('/volstore/spacebridge/gnssr_grzice/preprocessed_data/preprocessed_gnssr_update202330.csv', usecols=orig_feats)
data = data[orig_feats] # reorder columns
data = data.astype({orig_feats[-1]: 'float32'}) # we do not need greater than float32 precision for the label column
data.dropna(inplace=True)
orig_feats = data.keys()[2:-1] # original features are the features used for training; so no labels or data/time
data.reset_index(inplace= True, drop=True)
data

# %%
data.dtypes

# %%
print(f'Total memory used (Mb): {data.memory_usage().sum()/1e6}')

# %%
# get the indexes based on date/timeofyear queries to subset the data later for training and testing
datentime = data[['date','time']]#.drop(data.sample(frac=0.9,random_state=202206).index)
datentime['date'] = pd.to_datetime(data.date.astype('str'))
winterdatetime = datentime.query('20200501 <= date <= 20200930')
summerdatetime = datentime.query('date < 20200401 or date >= 20201101')
AprOctdatetime = datentime.drop(pd.concat([winterdatetime, summerdatetime]).index)
summertraindatetime = summerdatetime.sample(frac=0.9,random_state=202206)
summertestdatetime = summerdatetime.drop(summertraindatetime.index)
wintertraindatetime = winterdatetime.sample(frac=0.9,random_state=202206)
wintertestdatetime = winterdatetime.drop(wintertraindatetime.index)
AprOcttraindatetime = AprOctdatetime.sample(frac=0.9,random_state=202206)
AprOcttestdatetime = AprOctdatetime.drop(AprOcttraindatetime.index)
traindatetime = pd.concat([summertraindatetime, AprOcttraindatetime, wintertraindatetime]).sample(frac=1)
testdatetime = pd.concat([summertestdatetime, AprOcttestdatetime, wintertestdatetime])

# %%
# convert concentrations into presence/absence label 
palab = [1 if ic>0 else 0 for ic in data[data_lab_name[label]]]
data.drop(columns=[data_lab_name[label]], inplace=True) # change to the appropriate label column
data.drop(columns=['date', 'time'], inplace=True)
data['palab'] = palab

# train/test split
train_feat = data.loc[traindatetime.index if train_period=='allyear' else wintertraindatetime.index if train_period=='winter' else summertraindatetime.index if train_period=='summer' else None]
test_feat = data.loc[testdatetime.index]
winter_test_feat = data.loc[wintertestdatetime.index]
summer_test_feat = data.loc[summertestdatetime.index]

del data # memory management

# %%
# change the test param in pipeline to subset to 10% of training data for testing
if test:
  train_feat = train_feat.sample(frac=test_frac, random_state=202304)

train_feat

# %%
# set the folder to save the MLflow tracking output
mlflow.set_tracking_uri('/volstore/spacebridge/mlruns_archive/pycaret/mlruns')

# %%
# simple setup
simp_set = setup(data=train_feat, target='palab', 
                session_id=202304, 
                 transformation=False, normalize=True, train_size=0.8,
                numeric_features= list(orig_feats),
                 feature_interaction=False, 
                 feature_ratio=False,
                 polynomial_features = False,
                 feature_selection = False,
                feature_selection_threshold=0.8,
                 remove_multicollinearity = False, 
                 multicollinearity_threshold = 0.9,
                silent=True,
                log_experiment=False, experiment_name=f'PyCaret_lgbm_{test_frac}SimpData_{train_period}_{label}', log_plots=True,
                use_gpu=True,
                )

# %%
lgbm = create_model('lightgbm', fold=10)

# %%
# %%time
tuned_lgbm = tune_model(lgbm, optimize='MCC', fold=10)

# %%
save_model(tuned_lgbm, f'../products/PyCaret_lgbm_{test_frac}SimpData_{train_period}_{label}')

# %%
evaluate_model(tuned_lgbm)

# %%
plot_model(tuned_lgbm, plot="confusion_matrix")

# %%
plot_model(tuned_lgbm, plot="error")

# %%
plot_model(tuned_lgbm, plot="class_report")

# %%
# Columns used multicollinearity was removed
cols = get_config('X_train').columns
cols

# %%
test_feat[cols]

# %%
tuned_lgbm.score(test_feat[cols], test_feat.palab), tuned_lgbm.score(winter_test_feat[cols], winter_test_feat.palab), tuned_lgbm.score(summer_test_feat[cols], summer_test_feat.palab),

# %%
pred = tuned_lgbm.predict(test_feat[cols])
winterpred = tuned_lgbm.predict(winter_test_feat[cols])
summerpred = tuned_lgbm.predict(summer_test_feat[cols])

# %%
mcc(test_feat['palab'], pred), mcc(winter_test_feat['palab'], winterpred), mcc(summer_test_feat['palab'], summerpred), 
