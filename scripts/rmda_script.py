import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import minimize#, Bounds, LinearConstraint, NonlinearConstraint 
from bayes_opt import BayesianOptimization
import time
from datetime import datetime
import pickle


def mlefun(A,P,label,lab_names):
    K = P.shape[1]
    C = len(lab_names)
    A = np.exp(A)
    A = A.reshape((C,K))
    R = A/A.sum(axis=0)
    f = 0   
    for l,i in zip(lab_names, range(C)):
        f += np.sum(np.log( R[i] @ P[label==l].T )) 
    return -f

def mlefun_bayesian(**A):
    lab_names = ['YI_conc','FYI_conc','MYI_conc','water_conc']
    K = 4
    C = 4
    A = np.array([A['x'+str(i)] for i in range(C*K)])
    A = A.reshape((C,K))
    R = A/A.sum(axis=0)
    f = 0   
    for l,i in zip(lab_names, range(C)):
        f += np.sum(np.log( R[i] @ P[label==l].T )) 
    return f

if __name__ == '__main__':
    # load data
    # load preprocessed data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
    yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
    features = Xres.sample(frac=0.12, random_state=42)
    
    label = yres[features.index]
    lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
    lab_names = [lab for lab in lab_names if lab != 'ice_conc']

    # load umap transformation
    umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
    features = umap.transform(features)

    print(len(features), len(label))
    print(features[:5])
    print(label.head())

    # import bgmm object
    with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_SMOTERUS12pc_umap.pkl','rb') as f:
        bgmm = pickle.load(f)
    # predict probabilities
    P = bgmm.predict_proba(features)
    K = P.shape[1]
    C = len(lab_names) # number of classes; yi, myi, fyi, water
    # initial guess for scipy.optimize.minimize
    Ainit = np.log(np.array([np.sum(P[label==i], axis=0)/np.sum(label==i) for i in lab_names]))

    # define mle function for scipy.optimize.minimize
    tol = np.finfo(float).eps
    # surrogate function for bayesian optimization
    args = (P, label, lab_names)
    # bounds for bayesian optimization
    # bounds = {'x'+str(i): (0.0+np.finfo(float).eps, 10.) for i in range(C*K)}  
    bounds = tuple((0.0+tol,1.0-tol) for i in range(C*K))

    # # bayesian optimization
    # optimizer = BayesianOptimization(
    #     f = mlefun_bayesian,
    #     pbounds = bounds,
    #     # constraint = eq_constraint,
    #     random_state = 42,
    #     allow_duplicate_points=True,
    #     )

    # start = time.time()
    # N_ITER = 1000
    # optimizer.maximize(
    #     init_points = 20,
    #     n_iter = N_ITER,
    #     )
    # end = time.time()
    # print('time taken for bayesian optimization with '+str(N_ITER)+' iterations:', (end-start)/60., 'minutes') #around 
    # print(optimizer.max)
    # A_bayes = np.array([optimizer.max['params']['x'+str(i)] for i in range(C*K)]).reshape((C,K))
    # print('bayes opt result:', A_bayes/A_bayes.sum(axis=0))
    # # save bayes_opt result
    # method = "bayesian_opt"
    # timestamp = datetime.now().strftime('%Y%m%d:%H%M%S')
    # filename = '../products/models/rmda/rmda_umapfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+timestamp+'.pkl'
    # print(f'Saving file {filename}')
    # with open(filename,'wb') as f:
    #     pickle.dump(optimizer, f) 

    # method = 'COBYLA'
    # start = time.time()
    # result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
    # end = time.time()
    # print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
    # print(result)
    # A_scipy = np.exp(result.x).reshape((C,K))
    # print(method+' result:', A_scipy/A_scipy.sum(axis=0))
    # # save scipy result
    # timestamp = datetime.now().strftime('%Y%m%d:%H%M%S')
    # filename = '../products/models/rmda/rmda_umapfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+timestamp+'.pkl'
    # print(f'Saving file {filename}')
    # with open(filename,'wb') as f:
    #     pickle.dump(result, f)

    # method = 'BFGS'
    # start = time.time()
    # result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
    # end = time.time()
    # print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
    # print(result)
    # A_scipy = np.exp(result.x).reshape((C,K))
    # print(method+' result:', A_scipy/A_scipy.sum(axis=0))
    # # save scipy result
    # timestamp = datetime.now().strftime('%Y%m%d:%H%M%S')
    # filename = '../products/models/rmda/rmda_umapfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+timestamp+'.pkl'
    # print(f'Saving file {filename}')
    # with open('../products/models/rmda/rmda_umapfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    #     pickle.dump(result, f)

    # different initial guess for scipy.optimize.minimize
    # Ainit = np.array([0.25]*16).reshape((C,K))
    method = 'trust-constr'
    start = time.time()
    result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol, options={'verbose':2})
    end = time.time()
    print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
    print(result)
    A_scipy = np.exp(result.x).reshape((C,K))
    print(method+' result:', A_scipy/A_scipy.sum(axis=0))
    # save scipy result
    timestamp = datetime.now().strftime('%Y%m%d:%H%M%S')
    filename = '../products/models/rmda/rmda_umapfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+timestamp+'.pkl'
    print(f'Saving file {filename}')
    with open(filename,'wb') as f:
        pickle.dump(result, f)