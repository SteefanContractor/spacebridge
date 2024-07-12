import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import confusion_matrix
import pickle
import datetime

def build_clf(classifier, params, train_features, train_label, val_feats, val_label):
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

if __name__ == '__main__':
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
    yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
    train_features = Xres.sample(frac=0.12, random_state=42)
    train_label = yres.loc[train_features.index]
    print(train_label.value_counts())

    val_feats = train_features.sample(frac=0.1, random_state=42)
    val_label = train_label.loc[val_feats.index]
    val_label_int = val_label.astype('category').cat.codes
    train_features = train_features.drop(val_feats.index)
    train_label = train_label.drop(val_feats.index)
    train_label_int = train_label.astype('category').cat.codes

    # load umap transformation
    umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
    train_features = umap.transform(train_features)
    val_feats = umap.transform(val_feats)

    params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 4,
    'boosting': 'dart',
    'learning_rate': 0.1,
    'num_leaves': 63,
    'max_depth': 7,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 2500,
    'verbose': -1,
    }
    lgbm_clf = build_clf(LGBMClassifier, params, train_features=train_features, train_label=train_label, val_feats=val_feats, val_label=val_label)

    lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
    lab_names = [lab for lab in lab_names if lab != 'ice_conc']
    val_pred = lgbm_clf.predict(val_feats)
    # confusion matrix
    print('Label dictionary')
    # print(label_dict)
    print('LGBM confusion matrix')
    print(lab_names)
    print(confusion_matrix(val_label, val_pred, normalize='true', labels=lab_names))
    # confusion matrix normalized by predictions (columns)
    print(confusion_matrix(val_label, val_pred, normalize='pred', labels=lab_names))

    # confusion matrix normalised by only ice classes to compare with ice only model
    print(lab_names[:3])
    print(confusion_matrix(val_label, lgbm_clf.predict(val_feats), normalize='true', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))
    print(confusion_matrix(val_label, lgbm_clf.predict(val_feats), normalize='pred', labels=['YI_conc', 'FYI_conc', 'MYI_conc']))

    timestamp = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
    print(f'Saving lgbm_clf_4class_SMOTERUS12pc_umap_{timestamp}.pkl')   
    # save the models
    pickle.dump(lgbm_clf, open(f'../products/models/train_gradboost/lgbm_clf_4class_SMOTERUS12pc_umap_{timestamp}.pkl', 'wb'))
    # save params
    pickle.dump(params, open(f'../products/models/train_gradboost/lgbm_clf_4class_SMOTERUS12pc_umap_{timestamp}_params.pkl', 'wb'))

