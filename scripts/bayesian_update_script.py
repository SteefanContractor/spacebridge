import pandas as pd
import pickle
import numpy as np
from datetime import datetime

if __name__ == '__main__':
    # load preprocessed data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    lab_names =  ['YI_conc','FYI_conc','MYI_conc','water_conc']
    labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype='float32', usecols=['index']+lab_names)
    labels.index = labels.index.astype('int64')
    labels = labels[labels.water_conc < 100.]
    features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype='float64')
    features.index = features.index.astype('int64')
    features = features.loc[labels.index]
    print(len(features), len(labels))
    print(features.head())
    print(labels.head())

    # load lgbm model
    with open('../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_20240508:082525.pkl', 'rb') as f:
        lgbm_model = pickle.load(f)
    # load sk_bgmm model of rmda model
    with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl','rb') as f:
        bgmm_model = pickle.load(f)

    lgbm_confusion_mat = np.array([[0.67241379, 0.15922921, 0.04158215, 0.12677485],
                                [0.04790419, 0.84677703, 0.08559352, 0.01972526],
                                [0.046875,   0.25878906, 0.64257812, 0.05175781],
                                [0.01741254, 0.00994283, 0.00626461, 0.96638002]])
    R = np.array([[0.30267574, 0.01846785, 0.32744918, 0.15491719],
                [0.33155807, 0.00132933, 0.37493539, 0.0345478 ],
                [0.35329142, 0.00568585, 0.28904677, 0.15025457],
                [0.01247476, 0.97451697, 0.00856866, 0.66028044]])
    
    # predict proba lgbm
    # permutation according to lgbm_model.classes_ because labels are in the wrong order
    permutation = [int(np.where(lgbm_model.classes_ == l)[0]) for l in lab_names]
    start = datetime.now()
    lgbm_proba = lgbm_model.predict_proba(features)[:, permutation]
    end = datetime.now()
    print(f'Predicting lgbm_proba took {end-start}')
    # perdict proba bgmm
    start = datetime.now()
    bgmm_proba = bgmm_model.predict_proba(features)
    end = datetime.now()
    print(f'Predicting bgmm_proba took {end-start}')

    # lgbm
    print('Calculating lgbm posteriors')
    lgbm_post = (labels * (lgbm_confusion_mat @ lgbm_proba.T).T)
    lgbm_post = lgbm_post.div(lgbm_post.sum(axis=1), axis=0)
    # rmda
    print('Calculating rmda posteriors')
    rmda_post = (labels * (R @ bgmm_proba.T).T)
    rmda_post = rmda_post.div(rmda_post.sum(axis=1), axis=0)

    # save results
    timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
    print(f'Saving with timestamp: {timestamp}')
    lgbm_proba = pd.DataFrame(lgbm_proba, index=features.index, columns=lab_names)
    bgmm_proba = pd.DataFrame(bgmm_proba, index=features.index, columns=lab_names)
    lgbm_proba.to_csv(f'../products/results/lgbm_predict_proba_{timestamp}.csv')
    bgmm_proba.to_csv(f'../products/results/bgmm_predict_proba_{timestamp}.csv')
    lgbm_post.to_csv(f'../products/results/lgbm_posterior_proba_{timestamp}.csv')
    rmda_post.to_csv(f'../products/results/rmda_posterior_proba_{timestamp}.csv')