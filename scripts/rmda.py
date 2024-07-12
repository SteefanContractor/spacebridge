# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.mixture import BayesianGaussianMixture
from scipy.optimize import minimize, Bounds#, LinearConstraint, NonlinearConstraint 
from bayes_opt import BayesianOptimization
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from jax import grad, jacobian, hessian
import jax.numpy as jnp
# %%
# load preprocessed data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
features = Xres.sample(frac=0.01, random_state=42)
len(features)
# %%
label = yres[features.index]
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
lab_names = [lab for lab in lab_names if lab != 'ice_conc']
# lab_names = jnp.array(lab_names)
len(label)
# %%
# transform with umap
umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
features = umap.transform(features)
print(features[:5])
# %%
# import bgmm object
with open('../products/models/sk_bgmm/bgmm_4comp_1000iter_SMOTERUS12pc_umap.pkl','rb') as f:
    bgmm = pickle.load(f)
# %%
# predict probabilities
P = jnp.array(bgmm.predict_proba(features))
K = P.shape[1]
C = len(lab_names) # number of classes; yi, myi, fyi, water
# %%
# initial guess for scipy.optimize.minimize
Ainit_unbounded = jnp.log(jnp.array([jnp.sum(P[(label==i).to_numpy()], axis=0)/np.sum((label==i).to_numpy()) for i in lab_names]))
Ainit_bounded = jnp.array([jnp.sum(P[(label==i).to_numpy()], axis=0)/jnp.sum((label==i).to_numpy()) for i in lab_names])
# %%
# define mle function for scipy.optimize.minimize
tol = np.finfo(float).eps
def mlefun_unbounded(A,P,label,lab_names):
    K = P.shape[1]
    C = len(lab_names)
    A = jnp.exp(A)
    A = A.reshape((C,K))
    R = A/A.sum(axis=0)
    f = 0   
    for l,i in zip(lab_names, range(C)):
        f += jnp.sum(jnp.log( R[i] @ P[(label==l).to_numpy()].T ))
    return -f

def mlefun_bounded(A,P,label,lab_names):
    K = P.shape[1]
    C = len(lab_names)
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
bounds1 = tuple((0.0+tol,1.0-tol) for i in range(C*K))
bounds2 = Bounds([0-tol]*C*K, [1.0+tol]*C*K, keep_feasible=True)
# bounds for bayesian optimization
# bounds = {'x'+str(i): (0.0+np.finfo(float).eps, 10.) for i in range(C*K)}
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
result = minimize(fun=mlefun_bounded, x0=Ainit_bounded.flatten(), args=args, method=method, tol=tol, bounds=bounds1)
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
result = minimize(fun=mlefun_bounded, x0=Ainit_bounded.flatten(), args=args, method=method, tol=tol, bounds=bounds2)
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
method = 'trust-exact'
start = time.time()
result = minimize(fun=mlefun_unbounded, x0=Ainit_unbounded.flatten(), args=args, method=method, tol=tol, jac=jacobian(mlefun_unbounded), hess=hessian(mlefun_unbounded))
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
