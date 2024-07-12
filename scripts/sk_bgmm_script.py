import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV

# bic score
def gmm_bic_score(estimator, X):
    return estimator.score(X) - 0.5 * estimator.n_components * np.log(X.shape[0])

if __name__ == '__main__':
    # load data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
    features = Xres.sample(frac=0.12, random_state=42)
    # load umap transformation
    umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
    features = umap.transform(features)

    print(f'Data loaded with {len(features)} observations')

    # base model
    bgm = BayesianGaussianMixture(max_iter=1000, random_state=42)
    # set up grid search parameters
    param_grid = {'n_components': range(1,7)}
    # grid search
    grid_search = GridSearchCV(
        bgm, param_grid=param_grid, scoring=gmm_bic_score
    )
    start = datetime.now()
    grid_search.fit(features)
    end = datetime.now()
    print(f'Grid search took {end-start} seconds')

    print(grid_search.cv_results_)

    timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
    print(f'Saving grid search object with timestamp: {timestamp}')
    # save model
    pickle.dump(grid_search, open(f'../products/models/sk_bgmm/gridsearch_bgm_clean_SMOTERUS12pc_umap_{timestamp}.pkl', 'wb'))

    # define model
    start = datetime.now() 
    bgm = BayesianGaussianMixture(n_components=4, max_iter=1000, random_state=42)
    bgm.fit(features)
    end = datetime.now()   
    print(f'BGMM with 4 components on data with all features took {end-start} seconds')
    print(f'BIC score: {gmm_bic_score(bgm, features)}')
    print('converged' if bgm.converged_ else 'not converged')
    
    # save model
    if bgm.converged_:
        pickle.dump(bgm, open('../products/models/sk_bgmm/bgmm_4comp_1000iter_SMOTERUS12pc_umap.pkl', 'wb'))
    else:
        print('Model did not converge. Not saving model.')