# %%
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import confusion_matrix
import pickle
from datetime import datetime
import numpy as np
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
# load sk_bgmm model of rmda model
with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl','rb') as f:
    bgmm_model = pickle.load(f)
# load lgbm model
with open('../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_20240508:082525.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
# %%
with open('../products/models/rmda/rmda_allfeats_K4C4_trust-constr_20240512:012533.pkl', 'rb') as f:
        result = pickle.load(f)
R = result.x
R = np.exp(R)
R = R.reshape((4,4))
R = R/R.sum(axis=0)
print(R)
# %%
# RMDA predict function
def rmda_predict(R, bgmm_proba):
    # RMDA prediction
    rmda_proba = np.dot(R, bgmm_proba.T).T
    # RMDA prediction
    rmda_pred = np.argmax(rmda_proba, axis=1)
    return rmda_pred
# %%
# function to do lgbm and rmda prediction
def predict_labs(features, lgbm_model=lgbm_model, bgmm_model=bgmm_model, R=R, lab_names=lab_names):
    # predict proba lgbm
    start = datetime.now()
    lgbm_pred = lgbm_model.predict(features)
    end = datetime.now()
    print(f'Predicting lgbm_proba took {end-start}')
    # perdict proba bgmm
    start = datetime.now()
    bgmm_proba = bgmm_model.predict_proba(features)
    end = datetime.now()
    print(f'Predicting bgmm_proba took {end-start}')
    rmda_pred = rmda_predict(R, bgmm_proba)
    rmda_pred = [lab_names[i] for i in rmda_pred]
    return lgbm_pred, rmda_pred
# %%
print('training prediction')
train_lgbm_pred, train_rmda_pred = predict_labs(train_features)
print('validation prediction')
val_lgbm_pred, val_rmda_pred = predict_labs(val_feats)
print('test prediction')
test_lgbm_pred, test_rmda_pred = predict_labs(test_features)
# %%
# # Without UMAP
# ## Annual
# %%
# function to print confusion matrices
def print_confusion_matrices(label, lgbm_pred, rmda_pred, lab_names=lab_names):
    print('-'*20)
    print('all classes')
    print('-'*20)
    print('true normalized')
    print(lab_names)
    print('LGBM confusion matrix')
    print(confusion_matrix(label, lgbm_pred, normalize='true', labels=lab_names))
    print('RMDA confusion matrix')
    print(confusion_matrix(label, rmda_pred, normalize='true', labels=lab_names))
    print('-'*20)
    print('pred normalized')
    print(lab_names)
    print('LGBM confusion matrix')
    print(confusion_matrix(label, lgbm_pred, normalize='pred', labels=lab_names))
    print('RMDA confusion matrix')
    print(confusion_matrix(label, rmda_pred, normalize='pred', labels=lab_names))
    print('-'*20)
    print()
    print('-'*20)
    print('ice classes only')
    print('-'*20)
    print('true normalized')
    print(['YI_conc','FYI_conc','MYI_conc'])
    print('LGBM confusion matrix')
    print(confusion_matrix(label, lgbm_pred, normalize='true', labels=['YI_conc','FYI_conc','MYI_conc']))
    print('RMDA confusion matrix')
    print(confusion_matrix(label, rmda_pred, normalize='true', labels=['YI_conc','FYI_conc','MYI_conc']))
    print('-'*20)
    print('pred normalized')
    print(lab_names)
    print('LGBM confusion matrix')
    print(confusion_matrix(label, lgbm_pred, normalize='pred', labels=['YI_conc','FYI_conc','MYI_conc']))
    print('RMDA confusion matrix')
    print(confusion_matrix(label, rmda_pred, normalize='pred', labels=['YI_conc','FYI_conc','MYI_conc']))
    print('-'*20)
    print()
    print()
# %%
# ### Training
print_confusion_matrices(train_label, train_lgbm_pred, train_rmda_pred)
# %%
# ### Validation
print_confusion_matrices(val_label, val_lgbm_pred, val_rmda_pred)
# %%
# ### Testing
print_confusion_matrices(test_label, test_lgbm_pred, test_rmda_pred)
# %%
# save the RMDA test confusion matrix
test_cm = confusion_matrix(test_label, test_rmda_pred, normalize='true', labels=lab_names)
with open('../products/models/rmda/test_cm_rmda_allfeats_K4C4_trust-constr_20240512:012533.pkl', 'wb') as f:
    pickle.dump(test_cm, f)
# %%
# ## Seasonal
# %%
# summer (JFM)
# load metadata
metadata = pd.read_csv(data_path + "metadata.csv", index_col=0, parse_dates=['date'])
metadata = metadata.loc[labels.index]
winter = metadata.query('20200801 <= date <= 20201031')
summer = metadata.query('20200101 <= date <= 20200331') #date < 20200801 or date >= 20201101
shoulder = metadata.drop(pd.concat([winter, summer]).index)
assert len(winter)+len(summer)+len(shoulder) == len(metadata)
# %%
# filter train, val and test labels and feats by season
def filter_by_season(season, labels, lgbm_pred, rmda_pred):
    label_pred_df = pd.DataFrame({'label': labels, 'lgbm_pred': lgbm_pred, 'rmda_pred': rmda_pred}, index=labels.index)
    label_pred_df = label_pred_df.join(season, how='inner')[label_pred_df.columns]
    return label_pred_df.label, label_pred_df.lgbm_pred, label_pred_df.rmda_pred
# %%
# ### Training
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, train_label, train_lgbm_pred, train_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, train_label, train_lgbm_pred, train_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, train_label, train_lgbm_pred, train_rmda_pred))
# %%
# ### Validation
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, val_label, val_lgbm_pred, val_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, val_label, val_lgbm_pred, val_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, val_label, val_lgbm_pred, val_rmda_pred))
# %%
# ### Testing
# ### Validation
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, test_label, test_lgbm_pred, test_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, test_label, test_lgbm_pred, test_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, test_label, test_lgbm_pred, test_rmda_pred))
# %%
# # With UMAP
# ## Annual
# %%
# load sk_bgmm model of rmda model
with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_SMOTERUS12pc_umap.pkl','rb') as f:
    bgmm_model = pickle.load(f)
# load lgbm model
with open('../products/models/train_gradboost/lgbm_clf_4class_SMOTERUS12pc_umap_20240514:202900.pkl', 'rb') as f:
    lgbm_model = pickle.load(f)
# %%
with open('../products/models/rmda/rmda_umapfeats_K4C4_trust-constr_20240604:171926.pkl', 'rb') as f:
        result = pickle.load(f)
R = result.x
R = np.exp(R)
R = R.reshape((4,4))
R = R/R.sum(axis=0)
print(R)
# %%
# load umap transformation
with open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb") as f:
    umap = pickle.load(f)
# %%
print('training prediction')
train_lgbm_pred, train_rmda_pred = predict_labs(umap.transform(train_features))
print('validation prediction')
val_lgbm_pred, val_rmda_pred = predict_labs(umap.transform(val_feats))
print('test prediction')
test_lgbm_pred, test_rmda_pred = predict_labs(umap.transform(test_features))
# %%
# ### Training
print_confusion_matrices(train_label, train_lgbm_pred, train_rmda_pred)
# %%
# ### Validation
print_confusion_matrices(val_label, val_lgbm_pred, val_rmda_pred)
# %%
# ### Testing
print_confusion_matrices(test_label, test_lgbm_pred, test_rmda_pred)
# save the RMDA test confusion matrix
test_cm = confusion_matrix(test_label, test_rmda_pred, normalize='true', labels=lab_names)
with open('../products/models/rmda/test_cm_rmda_umapfeats_K4C4_trust-constr_20240604:171926.pkl', 'wb') as f:
    pickle.dump(test_cm, f)
# %%
# ### Training
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, train_label, train_lgbm_pred, train_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, train_label, train_lgbm_pred, train_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, train_label, train_lgbm_pred, train_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, train_label, train_lgbm_pred, train_rmda_pred))
# %%
# ### Validation
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, val_label, val_lgbm_pred, val_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, val_label, val_lgbm_pred, val_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, val_label, val_lgbm_pred, val_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, val_label, val_lgbm_pred, val_rmda_pred))
# %%
# ### Testing
# ### Validation
print('SUMMER')
print('_'*20)
print(filter_by_season(summer, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(summer, test_label, test_lgbm_pred, test_rmda_pred))
print('WINTER')
print('_'*20)
print(filter_by_season(winter, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(winter, test_label, test_lgbm_pred, test_rmda_pred))
print('SHOULDER')
print('_'*20)
print(filter_by_season(shoulder, test_label, test_lgbm_pred, test_rmda_pred)[0].value_counts())
print_confusion_matrices(*filter_by_season(shoulder, test_label, test_lgbm_pred, test_rmda_pred))