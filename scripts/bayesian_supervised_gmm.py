# Script to train a multivariate GMM in Pyro using SVI
# Does not converge in 1e6 steps
# Code adopted from https://gist.github.com/DanReia/e0456f9bd4cb35998d6029ffa31be60c
# and https://github.com/pyro-ppl/pyro/blob/dev/examples/lkj.py
# the issue is the guide I think
# couldn't get the MCMC to work either


# %%
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC

import matplotlib.pyplot as plt
# %%
def FiveGaussians():
    '''
    Data comes from Corduneanu and Bishop:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-aistats01.pdf
    '''
    
    means=torch.tensor([[0, 0], [3, -3], [3, 3], [-3, 3], [-3, -3]]).float()
    covariances=torch.tensor([[[1, 0],[0, 1]],[[1, 0.5], [0.5, 1]],[[1, -0.5], [-0.5, 1]],
                             [[1, 0.5],[0.5, 1]], [[1, -0.5],[-0.5, 1]]]).float()
    return dist.MultivariateNormal(means,covariances).sample([120]).view(-1,2)

data=FiveGaussians()
# %%
K=5

@config_enumerate(default='parallel')
def model(data):
    d = data.shape[1]

    pi = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))

    with pyro.plate('components', K):
        # Vector of variances for each of the d variables
        theta = dist.HalfCauchy(torch.ones(d)).rsample([K])
        # Lower cholesky factor of a correlation matrix
        concentration = torch.ones(
            ()
        )  # Implies a uniform distribution over correlation matrices
        L_omega = pyro.sample("L_omega", dist.LKJCholesky(d, concentration))
        # Lower cholesky factor of the covariance matrix
        T = torch.mm(theta.sqrt().diag_embed(), L_omega)

        means=data.mean(dim=0)
        scales=(0.5*torch.eye(data.size(1)))
        loc=pyro.sample('loc',dist.MultivariateNormal(means,scales))
        
    with pyro.plate('data', len(data)):
        assignment = pyro.sample('assignment', dist.Categorical(pi))
        pyro.sample('obs', dist.MultivariateNormal(loc[assignment],scale_tril=T[assignment]), obs=data)

# %% 
@config_enumerate(default="parallel")
def full_guide(data):
    with pyro.plate('data', len(data)):
        assignment_probs = pyro.param('assignment_probs', torch.ones(len(data), K) / K,
                                      constraint=constraints.unit_interval)
        pyro.sample('assignment', dist.Categorical(assignment_probs))
# %%
pyro.clear_param_store()

optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=1)
svi = SVI(model, full_guide, optim, loss=elbo)

pyro.set_rng_seed(42)
loss=[]
for i in range(10000):
    step_loss=svi.step(D)
    loss.append(step_loss)
# %%
plt.semilogx(loss)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()

print([i for i in pyro.get_param_store().items()])
# %%
# Try with MCMC
nuts_kernel = NUTS(model, jit_compile=False, step_size=1e-5)
MCMC(
    nuts_kernel,
    num_samples=200,
    warmup_steps=100,
    num_chains=4,
).run(D)
# %%
