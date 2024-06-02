# %%
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
# %%
# load preprocessed data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
features = Xres.sample(frac=0.65, random_state=42)
len(features)
# %%
label = yres[features.index]
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
lab_names = [lab for lab in lab_names if lab != 'ice_conc']
len(yres)
# %%
# # load preprocessed data
# data_path = '../data/'
# label = ['yi','fyi','myi']
# data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
#                      yi='YI_conc',
#                      myi='MYI_conc',
#                      fyi='FYI_conc')
# # read the header to get the column names
# col_names = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
# orig_feats = [col_names[i] for i in list(range(18))] 
# lab_names = [data_lab_name[l] for l in label]
# col_names = orig_feats + lab_names
# dtype = dict({c: 'float64' for c in col_names if c not in ['date']} , **{c: 'float32' for c in col_names if c in data_lab_name.values()})
# # # read values
# data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=col_names,dtype=dtype,parse_dates=['date'],low_memory=True)
# data = data[col_names] # reorder columns
# data[lab_names] = data[lab_names]/100.
# data['water_conc'] = 1. - data[lab_names].sum(axis=1)
# lab_names.append('water_conc')
# # drop -999.0 values from reflectivity1 and reflectivity2 columns
# data = data[data.reflectivity1!=-999.0]
# data = data[data.reflectivity2!=-999.0]
# data.dropna(inplace=True)
# orig_feats = orig_feats[2:-2] # original features are the features used for training; so no labels or data/time
# data.reset_index(inplace= True, drop=True)
# data
# # %%
# # log transform snr_reflected and power_reflected columns
# data['snr_reflected1'] = np.log10(data['snr_reflected1'])
# data['snr_reflected2'] = np.log10(data['snr_reflected2'])
# data['power_reflected1'] = np.log10(data['power_reflected1'])
# data['power_reflected2'] = np.log10(data['power_reflected2'])
# data['reflectivity1'] = np.log10(data['reflectivity1'])
# data['reflectivity2'] = np.log10(data['reflectivity2'])
# # drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
# data.dropna(inplace=True)# %%
# # %%
# # drop rows with excess_phase_noise2 < -5 (only 35 rows)
# data = data[data.excess_phase_noise2>-5.]
# # %%
# # min-max scale original features
# scaler = MinMaxScaler()
# data[orig_feats] = scaler.fit_transform(data[orig_feats])
# # %%
# # orig_data = data.copy()
# label = data[lab_names]
# label = label.idxmax(axis=1)
# # drop date, time, lat, lon columns
# data.drop(['date','time','latitude','longitude']+lab_names, axis=1, inplace=True)
# # %%
# # import PCA object
# with open('../products/models/sk_bgmm/pca_7comp.pkl','rb') as f:
#     pca = pickle.load(f)
# # %%
# # transform data
# data = pca.transform(data)
# %%
# import bgmm object
with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_allfeats.pkl','rb') as f:
    bgmm = pickle.load(f)
# %%
# predict probabilities
P = bgmm.predict_proba(features)
K = P.shape[1]
C = len(lab_names) # number of classes; yi, myi, fyi, water
# %%
# initial guess for scipy.optimize.minimize
Ainit = np.array([np.sum(P[label==i], axis=0)/np.sum(label==i) for i in lab_names])
# %%
# define mle function for scipy.optimize.minimize
tol = np.finfo(float).eps
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

# surrogate function for bayesian optimization
args = (P, label, lab_names)
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
    # return mlefun(R, *args)
# %%
# bounds = tuple((0.0+tol,1.0-tol) for i in range(C*K))
# bounds = Bounds([0-tol]*C*K, [1.0+tol]*C*K, keep_feasible=True)
# bounds for bayesian optimization
bounds = {'x'+str(i): (0.0+np.finfo(float).eps, 10.) for i in range(C*K)}
# %%
# constraints no longer necessary as they are reparamaterized in the objective function
# def A_linear_constraint(P, lab_names):
#     K = P.shape[1]
#     C = len(lab_names)
#     A = np.empty((K,C*K))
#     for i in range(K):
#         a = np.zeros((C,K))
#         a[:,i] = 1
#         A[i,:] = a.flatten()
#     return A
# A = A_linear_constraint(P, lab_names)
# eq_constraint = LinearConstraint(A, lb=[1.-tol]*K, ub=[1.+tol]*K)
# # nonlinear constraint for bayesian optimization
# def eq_constraint_fun(**R):
#     K = 2
#     C = 4
#     R = np.array([R['x'+str(i)] for i in range(C*K)])
#     R = R.reshape((C,K))
#     return R.sum(axis=0)
# eq_constraint = NonlinearConstraint(eq_constraint_fun, np.array([1.-tol]*K), np.array([1.+tol]*K))
# %%
# bayesian optimization
optimizer = BayesianOptimization(
    f = mlefun_bayesian,
    pbounds = bounds,
    # constraint = eq_constraint,
    random_state = 42,
    allow_duplicate_points=True,
    )

start = time.time()
N_ITER = 1000
optimizer.maximize(
    init_points = 20,
    n_iter = N_ITER,
    )
end = time.time()
print('time taken for bayesian optimization with '+str(N_ITER)+' iterations:', (end-start)/60., 'minutes') #around 
# %%
print(optimizer.max)
A_bayes = np.array([optimizer.max['params']['x'+str(i)] for i in range(C*K)]).reshape((C,K))
print('bayes opt result:', A_bayes/A_bayes.sum(axis=0))
# %%
# save bayes_opt result
method = "bayesian_opt"
with open('../products/models/rmda/rmda_allfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    pickle.dump(optimizer, f)
# %%
method = 'trust-constr'
start = time.time()
result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
end = time.time()
print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
# %%
print(result)
A_scipy = np.exp(result.x).reshape((C,K))
print(method+' result:', A_scipy/A_scipy.sum(axis=0))
# %%
# save scipy result
with open('../products/models/rmda/rmda_allfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    pickle.dump(result, f)
# %%
method = 'COBYLA'
start = time.time()
result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
end = time.time()
print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
# %%
print(result)
A_scipy = np.exp(result.x).reshape((C,K))
print(method+' result:', A_scipy/A_scipy.sum(axis=0))
# %%
# save scipy result
with open('../products/models/rmda/rmda_allfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    pickle.dump(result, f)
# %%
method = 'BFGS'
start = time.time()
result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
end = time.time()
print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
# %%
print(result)
A_scipy = np.exp(result.x).reshape((C,K))
print(method+' result:', A_scipy/A_scipy.sum(axis=0))
# %%
# save scipy result
with open('../products/models/rmda/rmda_allfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    pickle.dump(result, f)
# %%
# different initial guess for scipy.optimize.minimize
Ainit = np.array([0.25]*8).reshape((C,K))
# %%
method = 'BFGS'
start = time.time()
result = minimize(fun=mlefun, x0=Ainit.flatten(), args=args, method=method, tol=tol)
end = time.time()
print('time taken for '+method+' optimization:', (end-start)/60., 'minutes') #around 9 min
# %%
print(result)
A_scipy = np.exp(result.x).reshape((C,K))
print(method+' result:', A_scipy/A_scipy.sum(axis=0))
# %%
# save scipy result
with open('../products/models/rmda/rmda_allfeats_K'+str(K)+'C'+str(C)+'_'+method+'_'+str(datetime.now()).replace(' ','_')+'.pkl','wb') as f:
    pickle.dump(result, f)
# %%
