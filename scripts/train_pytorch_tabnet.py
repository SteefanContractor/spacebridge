# ### Note there are environmental clashes between pycaret and pytorch. Use environment.pytorch.yaml to create a separate conda environment and activate it before running this script.

# +
# Script to train tabnet written in pytorch. The pretraining section is work in progress.
# A separate script will be used for hyperparameter tuning with raytune or optuna

# + tags=["parameters"]
# declare a list tasks whose products you want to use as inputs
upstream = None


# +
import torch
# from torchmetrics.classification import BinaryMatthewsCorrCoef
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.metrics import Metric
import pandas as pd
import numpy as np
import mlflow
# import mlflow.pytorch
from mlflow.exceptions import MlflowException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef as mcc
import matplotlib.pyplot as plt

torch.cuda.is_available(), torch.cuda.get_device_name()
# -

import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

logger.info(f'Smoke testing: {test}, smoke_test data fraction: {test_frac}')

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

data.dtypes

data.memory_usage().sum()/1e6

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

# +
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

# +
# change the test param in pipeline to subset to 10% of training data for testing
if test:
  train_feat = train_feat.sample(frac=test_frac, random_state=202304)

train_feat

# +
val_feat = train_feat.sample(frac=0.1, random_state=202206)
train_feat = train_feat.drop(val_feat.index)

train_lab  = train_feat.iloc[:, -1]
train_feat = train_feat.iloc[:, :-1]
val_lab    = val_feat.iloc[:, -1]
val_feat   = val_feat.iloc[:, :-1]
# -

train_feat.shape, train_lab.shape, val_feat.shape, val_lab.shape

# +
# set up mlflow tracking
mlflow.set_tracking_uri('/volstore/spacebridge/mlruns_archive/pytorch/mlruns')

experiment_name = f"pytorch_tabnet_{label}_{train_period}"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

print(f'experiment id: {experiment_id}')


# -

class MCC(Metric):
    def __init__(self):
        self._name = "mcc"
        self._maximize = True

    def __call__(self, y_true, y_pred):
        print(y_pred[0])
        return mcc(y_true, np.argmax(y_pred))


with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    print(run_id)
    
    # Enable auto-logging to MLflow to capture TensorBoard metrics.
    mlflow.sklearn.autolog()
    
    # define model hyperparams
    # using all default parameters for now
    N_D=8, 
    N_A=8, 
    N_STEPS=3,   
    GAMMA=1.3, 
    N_INDEPENDENT=2, 
    N_SHARED=2,
    LAMBDA_SPARSE=1e-3, 
    MOMENTUM=0.02,
    OPTIMIZER=torch.optim.Adam,
    OPTIMIZER_PARAMS=dict(lr=2e-2)
    
    #fit params
    BATCH_SIZE=10240
    VIRTUAL_BATCH_SIZE=1024
    
    # log model params
    params = dict(n_d=N_D, n_a=N_A, n_steps=N_STEPS,   # using all default parameters for now
                  gamma=GAMMA, n_independent=N_INDEPENDENT, n_shared=N_SHARED,
                  lambda_sparse=LAMBDA_SPARSE, momentum=MOMENTUM,
                  optimizer_fn=OPTIMIZER,
                  optimizer_params=OPTIMIZER_PARAMS,
                  BATCH_SIZE=4096,
                  VIRTUAL_BATCH_SIZE=256
                 )
    mlflow.log_params(params)

    clf = TabNetClassifier(
                          #scheduler_params = {"gamma": 0.95,
                          #                 "step_size": 20},
                          #scheduler_fn=torch.optim.lr_scheduler.StepLR
                          )
                           
    clf.fit(train_feat.values, train_lab.values,
            eval_set=[(train_feat.values, train_lab.values), (val_feat.values, val_lab.values)],
            eval_name=['train','valid'],
            eval_metric=['accuracy','auc',],
            max_epochs=2,
            batch_size=BATCH_SIZE,
            virtual_batch_size=VIRTUAL_BATCH_SIZE)
    
    # mlflow.pytorch.log_model(clf, 'model')

# Time taken:  
#
# Batch_size | virtual_batch_size | time for 2 epochs(s)
# 4096         256                  (27, 59)
# 4096         128                  (39, 77)
# 2048         256                  (35, 70)
# 5120         512                  (22, 45)
# 10240        1024                 (17, 35) <---
# 20480        2048                 (16, 33)
# 40960        4096                 (16, 32)
# 10240        2048                 (16, 33)
# 20480        1024                 (18, 37)

# ## Pretraining

with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    print(run_id)
    
    # Enable auto-logging to MLflow to capture TensorBoard metrics.
    mlflow.sklearn.autolog()
    
    # define model hyperparams
    # using all default parameters for now
    N_D=8 
    N_A=8 
    N_STEPS=3   
    GAMMA=1.3 
    N_INDEPENDENT=2 
    N_SHARED=2
    LAMBDA_SPARSE=1e-3
    MOMENTUM=0.02
    OPTIMIZER=torch.optim.Adam
    OPTIMIZER_PARAMS=dict(lr=2e-2)
    SCHEDULER_PARAMS = {"gamma": 0.99,
                        "step_size": 100}
    SCHEDULER_FN=torch.optim.lr_scheduler.StepLR
    
    # pretrainer params
    PRETRAINING_RATIO=0.8
    MAX_EPOCHS=100
    BATCH_SIZE=10240
    VIRTUAL_BATCH_SIZE=1024
    
    pretrainparams = dict(max_epochs=MAX_EPOCHS,
                          batch_size=BATCH_SIZE,
                          virtual_batch_size=VIRTUAL_BATCH_SIZE,
                          pretraining_ratio=PRETRAINING_RATIO)
    #fit params
    MAX_EPOCHS=100
    BATCH_SIZE=10240
    VIRTUAL_BATCH_SIZE=1024
    FROM_UNSUPERVISED=True
    
    fitparams = dict(max_epochs=MAX_EPOCHS,
                  batch_size=BATCH_SIZE,
                  virtual_batch_size=VIRTUAL_BATCH_SIZE,
                     from_unsupervised=FROM_UNSUPERVISED
                    )
    
    # log model params
    modelparams = dict(n_d=N_D, n_a=N_A, n_steps=N_STEPS,   # using all default parameters for now
                  gamma=GAMMA, n_independent=N_INDEPENDENT, n_shared=N_SHARED,
                  lambda_sparse=LAMBDA_SPARSE, momentum=MOMENTUM,
                  optimizer_fn=OPTIMIZER,
                  optimizer_params=OPTIMIZER_PARAMS,
                  scheduler_params=SCHEDULER_PARAMS,
                  scheduler_fn=SCHEDULER_FN)
    
    
    
    mlflow.log_params(modelparams)
    mlflow.log_params(fitparams)
    
    if FROM_UNSUPERVISED:
      mlflow.log_params(pretrainparams)

      preclf = TabNetPretrainer(**modelparams)

      preclf.fit(X_train = train_feat.values, 
              eval_set=[train_feat.values, val_feat.values],
              eval_name=['train','valid'],
                 **pretrainparams
             )


      mlflow.sklearn.log_model(preclf, 'pretrainer')
    
    clf = TabNetClassifier(**modelparams)
    
    clf = clf.fit(X_train=train_feat.values, y_train=train_lab.values,
            eval_set=[(train_feat.values, train_lab.values), (val_feat.values, val_lab.values)],
            eval_name=['train','valid'],
            eval_metric=['accuracy','auc',],
                  **fitparams,
                 )
    
    
    mlflow.sklearn.log_model(clf, 'classifier')

val_feat.iloc[:2,:].values

plt.hist(val_feat['nasa_byu'], bins=40)

idx = val_feat.nasa_byu[val_feat.nasa_byu == 0.0].index.values

sfeats = val_feat.loc[idx,:].values
sfeats[:,14] = 0
sfeats

preclf.predict(sfeats)

# +
mlflow.pytorch.

clf.fit(
  train_feat.iloc[:, :-1].values, train_feat.iloc[:,-1].values,
  max_epochs=1,
  eval_metric=['accuracy']
)
# -

masked_val_feat = val_feat.copy()
masked_val_feat.iloc[:, -2:] = np.nan
masked_val_feat

clf.predict(masked_val_feat.iloc[:10,:].values)

mcc(val_lab[:1000].values, clf.predict(val_feat.iloc[:1000,:].values))

N = 100000
sum(clf.predict(train_feat.iloc[:N, :-1].values) == train_feat.iloc[:N, -1].values)/N

plt.figure(figsize=(15,8))
plt.bar(features, clf.feature_importances_)
plt.xticks(rotation=45)

explain_matrix, masks = clf.explain(train_feat.iloc[:1000, :-1].values)

# +
fig, axs = plt.subplots(1, 3, figsize=(20,10))

for i in range(3):
    axs[i].imshow(masks[i])
    axs[i].set_title(f"mask {i}")

