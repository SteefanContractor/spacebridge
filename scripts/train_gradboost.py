# %%
# import packages
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import confusion_matrix
import pickle
import datetime
# from skopt import BayesSearchCV
# from skopt.space import Categorical
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, ClusterCentroids, RepeatedEditedNearestNeighbours, AllKNN
# from imblearn.combine import SMOTEENN, SMOTETomek
# %%
data_path = "../data/preprocessed_gnssr_update202330_clean/"
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
train_features = Xres.sample(frac=0.12, random_state=42)
train_label = yres.loc[train_features.index]
display(train_label.value_counts())
# %%

# %%
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
# %%
nsamples = {lab: 100_000 for lab in lab_names[:-1]}
start = datetime.datetime.now()
smte = SMOTE(sampling_strategy='auto', random_state=42, n_jobs=8)
XSMOTE, ySMOTE = smte.fit_resample(train_features, train_label)
end = datetime.datetime.now()
print(f'Elapsed time: {end-start}')
display(ySMOTE.value_counts())
# # save resampled data
# XSMOTE.to_csv(data_path + "resampled/SMOTE_feats.csv")
# ySMOTE.to_csv(data_path + "resampled/SMOTE_labels.csv")
# %%
nsamples = {lab: 100_000 for lab in lab_names}
start = datetime.datetime.now()
rus = RandomUnderSampler(sampling_strategy=nsamples, random_state=42)
XSMOTERUS, ySMOTERUS = rus.fit_resample(XSMOTE, ySMOTE)
end = datetime.datetime.now()
print(f'Elapsed time: {end-start}')
display(ySMOTERUS.value_counts())
# %%
# now undersample the majority class
start = datetime.datetime.now()
enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority', max_iter=100, n_neighbors=3, n_jobs=10)
XSMOTEENN, ySMOTEENN = enn.fit_resample(XSMOTE, ySMOTE)
end = datetime.datetime.now()
print(f'Elapsed time: {end-start}')
display(ySMOTEENN.value_counts())
# save resampled data
XSMOTEENN.to_csv(data_path + "resampled/SMOTERENN_feats.csv")
ySMOTEENN.to_csv(data_path + "resampled/SMOTERENN_labels.csv")
# %%
# now undersample the majority class
start = datetime.datetime.now()
aknn = AllKNN(sampling_strategy='majority', n_jobs=10)
XSMOTEAKNN, ySMOTEAKNN = enn.fit_resample(XSMOTE, ySMOTE)
end = datetime.datetime.now()
print(f'Elapsed time: {end-start}')
display(ySMOTEAKNN.value_counts())
# save resampled data
XSMOTEAKNN.to_csv(data_path + "resampled/SMOTEAKNN_feats.csv")
ySMOTEAKNN.to_csv(data_path + "resampled/SMOTEAKNN_labels.csv")
# %%
# TomekLinks
start = datetime.datetime.now()
tl = TomekLinks(sampling_strategy='majority', n_jobs=10)
XTomek, yTomek = tl.fit_resample(XSMOTE, ySMOTE)
end = datetime.datetime.now()
print(f'Elapsed time: {end-start}')
display(yTomek.value_counts())
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
# train_features = XSMOTERUS
# train_label = ySMOTERUS
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
    start = datetime.datetime.now()
    clf.fit(train_features, train_label)
    end = datetime.datetime.now()
    print(f'Elapsed time: {end-start}')
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

for ne in [600, 700, 800, 900, 1000]:
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
    'n_estimators': 500,
    'verbose': -1,
}

for nl in [31, 41, 51, 61, 71]:
    params['num_leaves'] = nl
    print(f'num_leaves: {nl}')
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
    'learning_rate': 0.1,
    'num_leaves': 95,
    'max_depth': 7,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 5000,
    'verbose': -1,
}
lgbm_clf = build_clf(LGBMClassifier, params)
# %%
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
lab_names = [lab for lab in lab_names if lab != 'ice_conc']
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
timestamp = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
print(f'Saving lgbm_clf_4class_cleanSMOTEdata_{timestamp}.pkl')   
# save the models
pickle.dump(lgbm_clf, open(f'../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_{timestamp}.pkl', 'wb'))
# save params
pickle.dump(params, open(f'../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_params_{timestamp}.pkl', 'wb'))
# %%
# Now the hyperparameter tuning is done, we can use the whole training set to train the model
# assign the rest for training
train_label = label_encoded.drop(test_label.index)
# randomly sample 1% of water_conc to balance training set
water_conc = train_label[train_label == 'water_conc'].sample(frac=0.01, random_state=42)
train_label = train_label[train_label != 'water_conc'].append(water_conc)
display(train_label.value_counts())
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
train_features = features.loc[train_label.index]
test_features = features.loc[test_label.index]
# %%
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
lgbm_clf = LGBMClassifier(**params)
lgbm_clf.fit(train_features, train_label)
# %%
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
test_label.value_counts()
# %%
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
# train_features = features.loc[train_label.index]
test_features = features.loc[test_label.index]
#############################################################################
#############################################################################
#############################################################################
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
# put 10% aside for testing
test_label = label_encoded.sample(frac=0.1, random_state=42)
display(test_label.value_counts())
# %%
# load the features but read only the training and testing indices
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
test_features = features.loc[test_label.index]
# %%
# load val dataset
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
train_features = Xres.sample(frac=0.12, random_state=42)
train_label = yres.loc[train_features.index]
display(train_label.value_counts())
# %%
val_feats = train_features.sample(frac=0.1, random_state=42)
val_label = train_label.loc[val_feats.index]
val_label_int = val_label.astype('category').cat.codes
train_features = train_features.drop(val_feats.index)
train_label = train_label.drop(val_feats.index)
train_label_int = train_label.astype('category').cat.codes
# %%
# load umap transformation
with open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb") as f:
    umap = pickle.load(f)
# train_features = umap.transform(train_features)
val_feats = umap.transform(val_feats)
test_features = umap.transform(test_features)
# %%
# load lgbm model
lgbm_clf = pickle.load(open('../products/models/train_gradboost/lgbm_clf_4class_SMOTERUS12pc_umap_20240514:202900.pkl', 'rb'))
# %%
train_score = lgbm_clf.score(train_features, train_label)
print(f'Train score: {train_score}')
# %%
# val score
print(f'Validation score: {lgbm_clf.score(val_feats, val_label)}')
# test score
print(f'Test score: {lgbm_clf.score(test_features, test_label)}')
# %%
val_pred = lgbm_clf.predict(val_feats)
# confusion matrix
print('Validation data LGBM confusion matrix')
print(lab_names)
print(confusion_matrix(val_label, val_pred, labels=lab_names))
# %%
# confusion matrix normalised by predictions (columns)
print('Normalised by true labels')
print(lab_names)
print(confusion_matrix(val_label, val_pred, normalize='true', labels=lab_names))
print('Noralised by pred labels')
print(lab_names)
print(confusion_matrix(val_label, val_pred, normalize='pred', labels=lab_names))
# %%
# confusion matrix normalised by only ice classes to compare with ice only model
print('Normalised by true labels')
print(lab_names[:3])
print(confusion_matrix(val_label, val_pred, normalize='true', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
print('Normalised by pred labels')
print(lab_names[:3])
print(confusion_matrix(val_label, val_pred, normalize='pred', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
# %%
# confusion matrix
test_pred = lgbm_clf.predict(test_features)
print('Label dictionary')
# print(label_dict)
print('LGBM confusion matrix')
print('Normalised by true labels')
print(lab_names)
print(confusion_matrix(test_label, test_pred, normalize='true', labels=lab_names))
# confusion matrix normalized by predictions (columns)
print('Normalised by pred labels')
print(lab_names[:3])
print(confusion_matrix(test_label, test_pred, normalize='pred', labels=lab_names))
# %%
# confusion matrix normalised by only ice classes to compare with ice only model
print('Normalised by true labels')
print(lab_names[:3])
print(confusion_matrix(test_label, test_pred, normalize='true', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
print('Normalised by pred labels')
print(lab_names[:3])
print(confusion_matrix(test_label, test_pred, normalize='pred', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
# %%
# unnormalized confusion matrix
print('LGBM confusion matrix')
print(lab_names)
print(confusion_matrix(test_label, test_pred, labels=lab_names))
# %%
# save test confusion matrix normalised by true labels
test_cm_true = confusion_matrix(test_label, test_pred, normalize='true', labels=lab_names)
with open('../products/models/train_gradboost/test_cm_true_lgbm_clf_4class_SMOTERUS12pc_umap_20240514:202900.pkl', 'wb') as f:
    pickle.dump(test_cm_true, f)
# %%
