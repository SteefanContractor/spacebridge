import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import argparse

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Bayesian Update Script')
    parser.add_argument('--umap', action='store_true', help='Flag to indicate whether to use UMAP transformation')
    args = parser.parse_args()

    # load preprocessed data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    lab_names =  ['YI_conc','FYI_conc','MYI_conc','water_conc']
    labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype='float32', usecols=['index']+lab_names)
    labels.index = labels.index.astype('int64')
    labels = labels[labels.water_conc < 100.]
    features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype='float64')
    features.index = features.index.astype('int64')
    features = features.loc[labels.index]
    features_index = features.index
    print(len(features), len(labels))
    print(features.head())
    print(labels.head())

    if args.umap:
        #umap transformation of features
        start = datetime.now()
        umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
        features = umap.transform(features)
        end = datetime.now()
        print(f"UMAP transformation took {end-start} seconds")
        print(f'Num umap features: {features.shape}')

        # # load lgbm model
        # with open('../products/models/train_gradboost/lgbm_clf_4class_SMOTERUS12pc_umap_20240514:202900.pkl', 'rb') as f:
        #     lgbm_model = pickle.load(f)
        # load sk_bgmm model of rmda model
        with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_SMOTERUS12pc_umap.pkl','rb') as f:
            bgmm_model = pickle.load(f)
        # # load lgbm confusion matrix
        # with open('../products/models/train_gradboost/test_cm_true_lgbm_clf_4class_SMOTERUS12pc_umap_20240514:202900.pkl', 'rb') as f:
        #     lgbm_confusion_mat = pickle.load(f)
        # load rmda confusion matrix
        with open('../products/models/rmda/test_cm_rmda_umapfeats_K4C4_trust-constr_20240604:171926.pkl', 'rb') as f:
            rmda_confusion_mat = pickle.load(f)
        # load rmda transformation matrix
        with open('../products/models/rmda/rmda_umapfeats_K4C4_trust-constr_20240604:171926.pkl', 'rb') as f:
            result = pickle.load(f)
        R = result.x
        R = np.exp(R)
        R = R.reshape((4,4))
        R = R/R.sum(axis=0)
        print(R)
    else:
        # # load lgbm model
        # with open('../products/models/train_gradboost/lgbm_clf_4class_cleanSMOTEdata_20240508:082525.pkl', 'rb') as f:
        #     lgbm_model = pickle.load(f)
        # load sk_bgmm model of rmda model
        with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl','rb') as f:
            bgmm_model = pickle.load(f)
        # # load lgbm confusion matrix
        # with open('../products/models/train_gradboost/test_cm_true_lgbm_clf_4class_cleanSMOTEdata_20240508:082525.pkl', 'rb') as f:
        #     lgbm_confusion_mat = pickle.load(f)
        # load rmda confusion matrix
        with open('../products/models/rmda/test_cm_rmda_allfeats_K4C4_trust-constr_20240512:012533.pkl', 'rb') as f:
            rmda_confusion_mat = pickle.load(f)
        # load rmda transformation matrix
        with open('../products/models/rmda/rmda_allfeats_K4C4_trust-constr_20240512:012533.pkl', 'rb') as f:
            result = pickle.load(f)
        R = result.x
        R = np.exp(R)
        R = R.reshape((4,4))
        R = R/R.sum(axis=0)
    
    # predict proba lgbm
    # # permutation according to lgbm_model.classes_ because labels are in the wrong order
    # permutation = [int(np.where(lgbm_model.classes_ == l)[0]) for l in lab_names]
    # start = datetime.now()
    # lgbm_proba = lgbm_model.predict_proba(features)[:, permutation]
    # end = datetime.now()
    # print(f'Predicting lgbm_proba took {end-start}')
    # perdict proba bgmm
    start = datetime.now()
    bgmm_proba = bgmm_model.predict_proba(features)
    rmda_proba = R @ bgmm_proba.T
    end = datetime.now()
    print(f'Predicting bgmm_proba took {end-start}')
    assert np.allclose(rmda_proba.sum(axis=0), 1)

    # # lgbm
    # print('Calculating lgbm posteriors')
    # lgbm_post = (labels * (lgbm_confusion_mat @ lgbm_proba.T).T)
    # lgbm_post = lgbm_post.div(lgbm_post.sum(axis=1), axis=0)
    # rmda
    print('Calculating rmda posteriors')
    rmda_post = (labels * (rmda_confusion_mat @ rmda_proba).T)
    rmda_post = rmda_post.div(rmda_post.sum(axis=1), axis=0)

    # save results
    timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
    print(f'Saving with timestamp: {timestamp}')
    if args.umap:
        # lgbm_proba = pd.DataFrame(lgbm_proba, index=features_index, columns=lab_names)
        # bgmm_proba = pd.DataFrame(bgmm_proba, index=features_index, columns=lab_names)
        # lgbm_proba.to_csv(f'../products/results/umap_lgbm_predict_proba_{timestamp}.csv')
        # bgmm_proba.to_csv(f'../products/results/umap_bgmm_predict_proba_{timestamp}.csv')
        # lgbm_post.to_csv(f'../products/results/umap_lgbm_posterior_proba_{timestamp}.csv')
        rmda_post.to_csv(f'../products/results/umap_rmda_posterior_proba_{timestamp}.csv')
    else:
        # lgbm_proba = pd.DataFrame(lgbm_proba, index=features_index, columns=lab_names)
        # bgmm_proba = pd.DataFrame(bgmm_proba, index=features_index, columns=lab_names)
        # lgbm_proba.to_csv(f'../products/results/umap_lgbm_predict_proba_{timestamp}.csv')
        # bgmm_proba.to_csv(f'../products/results/umap_bgmm_predict_proba_{timestamp}.csv')
        # lgbm_post.to_csv(f'../products/results/umap_lgbm_posterior_proba_{timestamp}.csv')
        rmda_post.to_csv(f'../products/results/withoutumap_rmda_posterior_proba_{timestamp}.csv')