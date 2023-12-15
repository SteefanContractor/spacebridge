# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from collections import defaultdict
import torch
import numpy as np
import scipy.stats
from torch.distributions import constraints
from matplotlib import pyplot

%matplotlib inline

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
# %%
# load preprocessed data
data_path = '../data/'
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))] 
dtype = {c: 'float64' for c in orig_feats if c not in ['date']}
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[orig_feats] # reorder columns
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = data.keys()[2:-2] # original features are the features used for training; so no labels or data/time
data.reset_index(inplace= True, drop=True)
data
# %%
# log transform snr_reflected and power_reflected columns
data['snr_reflected1'] = np.log10(data['snr_reflected1'])
data['snr_reflected2'] = np.log10(data['snr_reflected2'])
data['power_reflected1'] = np.log10(data['power_reflected1'])
data['power_reflected2'] = np.log10(data['power_reflected2'])
data['reflectivity1'] = np.log10(data['reflectivity1'])
data['reflectivity2'] = np.log10(data['reflectivity2'])
# drop NaN values introduced from log (only 161 reflectivity1 and 32 reflectivity2 values are <= 0)
data.dropna(inplace=True)# %%
# %%
# drop rows with excess_phase_noise2 < -5 (only 35 rows)
data = data[data.excess_phase_noise2>-5.]
# %%
# min-max scale original features
scaler = MinMaxScaler()
data[orig_feats] = scaler.fit_transform(data[orig_feats])
# %%
orig_data = data.copy()
# drop date, time, lat, lon columns
data.drop(['date','time','latitude','longitude'], axis=1, inplace=True)
# %%
# convert to tensor
data = torch.tensor(data.values)
# %%
data = torch.tensor([[0.0,5.0], [1.0,1.0], [10.0,3.0], [11.0,10.0], [12.0,0.0]])
# %%
K = 4  # Fixed number of components. Same as number of labelled classes

@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    diag = pyro.sample("diag", dist.LogNormal(0.0, 2.0).expand([K, data.shape[1]]))
    with pyro.plate("components", K):
        locs = pyro.sample("locs", dist.Normal(0.0, 10.0))
        # Define the covariance matrix.
        corr_tril = pyro.sample("corr_tril", dist.LKJCholesky(2, K))
        cov_matrix = torch.matmul(corr_tril, torch.matmul(torch.diag_embed(diag), corr_tril.transpose(-1, -2)))
        pyro.param("cov_matrix", cov_matrix)
    with pyro.plate("data", len(data)):
        # Local variables.
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        pyro.sample("obs", dist.MultivariateNormal(locs[assignment], cov_matrix[assignment]), obs=data)
# %%
K = 4  # Fixed number of components. Same as number of labelled classes

@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample("weights", dist.Dirichlet(0.5 * torch.ones(K)))
    prec_matrix = pyro.sample("prec_matrix", dist.Wishart(4, torch.eye(data.shape[1])).expand([K]))
    with pyro.plate("components", K):
        locs = pyro.sample("locs", dist.Normal(0.0, 10.0))
    with pyro.plate("data", len(data)):
        # Local variables.
        assignment = pyro.sample("assignment", dist.Categorical(weights))
        pyro.sample("obs", dist.MultivariateNormal(locs[assignment], prec_matrix[assignment]), obs=data)


# %%
optim = pyro.optim.Adam({"lr": 0.1, "betas": [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
# calculate the inverse of the data covariance
# data_cov_inv = torch.inverse(torch.tensor(data.cov().values))
def init_loc_fn(site):
    if site["name"] == "weights":
        # Initialize weights to uniform.
        return torch.ones(K) / K
    if site["name"] == "prec_matrix":
        return data_cov_inv
    if site["name"] == "locs":
        return data[torch.multinomial(torch.ones(len(data)) / len(data), K)]
    raise ValueError(site["name"])


def initialize(seed):
    global global_guide, svi
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    global_guide = AutoDelta(
        poutine.block(model, expose=["weights", "locs", "prec_matrix"]),
        # init_loc_fn=init_loc_fn,
    )
    svi = SVI(model, global_guide, optim, loss=elbo)
    return svi.loss(model, global_guide, data)


# Choose the best among 100 random initializations.
loss, seed = min((initialize(seed), seed) for seed in range(100))
initialize(seed)
print(f"seed = {seed}, initial_loss = {loss}")
# %%
