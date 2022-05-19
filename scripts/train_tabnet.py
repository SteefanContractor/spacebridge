# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
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
upstream = ['preprocess_gnssr']

# This is a placeholder, leave it as None
product = None


# %%
# load libraries
import tensorflow as tf
from tensorflow.keras import *
import mlflow
from mlflow.exceptions import MlflowException
import tabnet

# %%
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% tags=["dataset-load"]
# load and import data
# only need to run this cell once
BATCH_SIZE=2048
gnssr_csv_path = '/volstore/spacebridge/gnssr_grzice/preprocessed_gnssr.csv'
tfd_gnssr = tf.data.experimental.make_csv_dataset(gnssr_csv_path,
                                                  batch_size=BATCH_SIZE,
                                                  num_epochs=1,
                                                  shuffle=True,
                                                  shuffle_buffer_size=10000,
                                                  shuffle_seed=202205,
                                                  sloppy=False,
                                                  label_name='MYI_conc',
                                                  select_columns=list(range(32))+[34],
                                                  ignore_errors=False,)

# tfd_gnssr = tfd_gnssr.cache()

n = sum(1 for _ in tfd_gnssr)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)
n_test = n - n_train - n_valid

train_ds = tfd_gnssr.take(n_train)
valid_ds = tfd_gnssr.skip(n_train).take(n_valid)
test_ds = tfd_gnssr.skip(n_train + n_valid).take(n_test)

# %%
# print first five elements of all features and labels of a single batch and define a feature_column vector
feature_columns = []
for batch, label in train_ds.take(1):
  for key, value in batch.items():
    print(f"{key:20s}: {value[:5]}")
    feature_columns.append(tf.feature_column.numeric_column(key))
  print()
  print(f"{'label':20s}: {label[:5]}")

# %%
# cache dataset for speed 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
valid_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
# set up mlflow tracking
mlflow.set_tracking_uri('/home/stefan/GIT_REPOS/spacebridge/mlruns')

experiment_name = "tabnet_exploratory"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

print(f'experiment id: {experiment_id}')

# %%
# start logging
with mlflow.start_run(experiment_id=experiment_id) as run:
    print(run.info.run_id)
    
    # Enable auto-logging to MLflow to capture TensorBoard metrics.
    mlflow.tensorflow.autolog()
    
    # define model hyperparams
    # adjust batch_size param in "dataset-load" cell
    FEATURE_DIM=32
    OUTPUT_DIM=30
    NUM_DECISION_STEPS=6
    RELAXATION_FACTOR=1.0
    SPARCITY_COEFFICIENT=1e-5
    BATCH_MOMENTUM=0.98
    VIRTUAL_BATCH_SIZE=None
    NORM_TYPE='group'
    NUM_GROUPS=1
    
    # log model params
    params = {'batch_size': BATCH_SIZE,
            'feature_dim': FEATURE_DIM,
            'output_dim': OUTPUT_DIM,
            'num_decision_steps': NUM_DECISION_STEPS,
            'relaxation_factor': RELAXATION_FACTOR,
            'sparcity_coefficient': SPARCITY_COEFFICIENT,
            'batch_momentum': BATCH_MOMENTUM,
            'virtual_batch_size': VIRTUAL_BATCH_SIZE,
            'norm_type': NORM_TYPE,
            'num_groups': NUM_GROUPS}
    mlflow.log_params(model_params)
    
    # specify tabnet model
    model = tabnet.TabNetRegressor(feature_columns,
                                num_regressors=1,
                                feature_dim=FEATURE_DIM, 
                                output_dim=OUTPUT_DIM,
                                num_decision_steps=NUM_DECISION_STEPS, 
                                relaxation_factor=RELAXATION_FACTOR,
                                sparsity_coefficient=SPARCITY_COEFFICIENT,
                                batch_momentum=BATCH_MOMENTUM,
                                virtual_batch_size=VIRTUAL_BATCH_SIZE, 
                                norm_type=NORM_TYPE,
                                num_groups=NUM_GROUPS)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100, decay_rate=0.8, staircase=False)
    optimizer = tf.keras.optimizers.Adam(lr)#learning_rate=0.001
    model.compile(optimizer, loss='mse', metrics=['accuracy'])

    model.fit(train_ds, epochs=500, validation_data=valid_ds, verbose=2)

# %%
