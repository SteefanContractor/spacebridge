# %%
# import packages
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
import pickle
import datetime
from skopt import BayesSearchCV
from skopt.space import Categorical
# %%
data_path = "../data/preprocessed_gnssr_update202330_clean/"
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
dtype = {label: 'float32' for label in lab_names}
labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype=dtype)
# subset to sure labels
labels = labels[(labels.YI_conc > 90.) | (labels.FYI_conc > 99.9) | (labels.MYI_conc > 99.) | (labels.water_conc == 100.)]
# drop ice_conc column
labels.drop('ice_conc', axis=1, inplace=True)
lab_names = [lab for lab in lab_names if lab != 'ice_conc']
label_encoded = labels.idxmax(axis=1)
# count the number of each label
label_encoded.value_counts()
# allocate 1000 samples of each label for testing
# test = labels.groupby(label_encoded).apply(lambda x: x.sample(1000, random_state=42))
# put 10% aside for testing
test_label = label_encoded.sample(frac=0.1, random_state=42)
display(test_label.value_counts())
# assign the rest for training
train_label = label_encoded.drop(test_label.index)
# randomly sample 1% of water_conc to balance training set
water_conc = train_label[train_label == 'water_conc'].sample(frac=0.01, random_state=42)
train_label = train_label[train_label != 'water_conc'].append(water_conc)
display(train_label.value_counts())
# %%
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
train_features = features.loc[train_label.index]
test_features = features.loc[test_label.index]
# %%
# bayesian optimization based hyperparameter tuning fails because of outdated packages (sklearn)
# not necessary anyway as gridsearch is fast enough
# params = {
#     'objective': 'multiclass',
#     'metric': 'multi_logloss',
#     'num_class': 4,
#     # 'boosting': 'gbdt',
#     'learning_rate': 0.05,
#     # 'num_leaves': 31,
#     'verbose': -1,
# }
# # bayesian optimization based hyperparameter tuning
# opt = BayesSearchCV(
#     LGBMClassifier(**params),
#     {
#         'num_leaves': Categorical([31, 63, 127, 255, 511, 1023]),
#         'boosting': Categorical(['gbdt', 'dart']),
#         # 'learning_rate': (0.01, 0.1),
#         # 'n_estimators': (100, 1000),
#         # 'max_depth': (3, 10),
#     },
#     n_iter=32,
#     cv=5,
#     n_jobs=8,
#     verbose=1,
#     random_state=42
# )
# opt.fit(train_features, train_label)
# save optimizer
# pickle.dump(opt, open(f'../products/models/train_gradboost/lgbm_bayesoptcv_{datetime.date.today()}.pkl', 'wb'))
# %%
val_feats = train_features.sample(frac=0.1, random_state=42)
val_label = train_label.loc[val_feats.index]
val_label_int = val_label.astype('category').cat.codes
train_features = train_features.drop(val_feats.index)
train_label = train_label.drop(val_feats.index)
train_label_int = train_label.astype('category').cat.codes
def build_clf(classifier, params, train_features=train_features, train_label=train_label, val_feats=val_feats, val_label=val_label):
    # use above as kwargs for lgbmclassifier
    clf = classifier(**params)
    # fit the model
    clf.fit(train_features, train_label)
    # train/val score
    train_score = clf.score(train_features, train_label)
    val_score = clf.score(val_feats, val_label)
    print(f'Train score: {train_score}')
    print(f'Validation score: {val_score}')
    print(f'overfitting normalised by number of estimators: {(train_score-val_score)/params["n_estimators"]}')
    return clf
# %%
# set lgbm parameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'boosting': 'dart',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 200,
    'verbose': -1,
}

for ne in [100, 200, 300, 400, 500]:
    params['n_estimators'] = ne
    print(f'n_estimators: {ne}')
    build_clf(LGBMClassifier, params)
# %%
# set lgbm parameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'boosting': 'dart',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 200,
    'verbose': -1,
}

for ne in [500, 600, 700]:
    params['n_estimators'] = ne
    print(f'n_estimators: {ne}')
    build_clf(LGBMClassifier, params)
# %%
# xgboost overfitts even more while being slower
# # set xgb parameters
# params = {
#     'objective': 'multi:softprob',
#     'num_class': 4,
#     'booster': 'gbtree',
#     'learning_rate': 0.05,
#     'max_depth': 6,

# }

# for md in [6, 7, 8, 9]:
#     params['max_depth'] = md
#     print(f'max_depth: {md}')
#     build_clf(XGBClassifier, params, train_label=train_label_int, val_label=val_label_int)

# %%
# overfitting increases beyond max_depth=6 and num_leaves=31
# pick max_depth=6 and num_leaves=31
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'boosting': 'dart',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 500,
    'verbose': -1,
}
lgbm_clf = build_clf(LGBMClassifier, params)
# %%
# confusion matrix
print('Label dictionary')
# print(label_dict)
print('LGBM confusion matrix')
print(lab_names)
print(confusion_matrix(val_label, lgbm_clf.predict(val_feats), normalize='true', labels=lab_names))
# confusion matrix normalized by predictions (columns)
print(confusion_matrix(val_label, lgbm_clf.predict(val_feats), normalize='pred', labels=lab_names))
# %%
# confusion matrix normalised by only ice classes to compare with ice only model
print(lab_names[:3])
print(confusion_matrix(val_label, lgbm_clf.predict(val_feats), normalize='pred', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
# %%
timestamp = datetime.datetime.now()   
# save the models
pickle.dump(lgbm_clf, open(f'../products/models/train_gradboost/lgbm_clf_4class_cleandata_{timestamp}.pkl', 'wb'))
# save params
pickle.dump(params, open(f'../products/models/train_gradboost/lgbm_clf_4class_cleandata_params_{timestamp}.pkl', 'wb'))
# %%
