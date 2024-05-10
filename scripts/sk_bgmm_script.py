import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import pickle
import time

# bic score
def gmm_bic_score(estimator, X):
    return estimator.score(X) - 0.5 * estimator.n_components * np.log(X.shape[0])

if __name__ == '__main__':
    # load data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
    features = Xres.sample(frac=0.65, random_state=42)
    print(f'Data loaded with {len(features)} observations')

    # define model
    start = time.time() 
    bgm = BayesianGaussianMixture(n_components=4, max_iter=1000, random_state=42)
    bgm.fit(Xres)
    end = time.time()   
    print(f'BGMM with 4 components on data with all features took {end-start} seconds')
    print(f'BIC score: {gmm_bic_score(bgm, Xres)}')
    print('converged' if bgm.converged_ else 'not converged')
    # %%
    # save model
    if bgm.converged_:
        pickle.dump(bgm, open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl', 'wb'))
    else:
        print('Model did not converge. Not saving model.')