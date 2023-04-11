# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# This script takes the preprocessed_gnssr.csv
# and adds the iceage concentration data from osisaf dataset along with Nasa BYU scatterometer data as an auxiliary variable.
# 
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# %load_ext autoreload
# %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = None

# This is a placeholder, leave it as None
product = None


# %%
import pandas as pd
# import dask.dataframe as dd
import numpy as np
# import matplotlib.pyplot as plt
import datetime
import xarray as xr
from scipy.spatial import cKDTree
import os
# import cartopy.crs as ccrs

# %%
import sys
sys.path.append('/home/stefan/GIT_REPOS/spacebridge/pkgs')
# from importlib import reload
# import data_wrangling.gnssr
# reload(data_wrangling.gnssr)
from data_wrangling.gnssr import create_iam_label_columns, create_oi_label_columns, create_byu_feat_columns, latlon_to_ijidx

# %%
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# %%
# your code here...
data = pd.read_csv('/volstore/spacebridge/gnssr_grzice/preprocessed_data/preprocessed_gnssr.csv', usecols=list(range(16))+[30,31,32,33,34])
orig_feats = data.keys()[:-3]
data = data.dropna(subset=orig_feats) # do not drop rows with nans in the labels
data.reset_index(inplace= True, drop=True)
data

# %%
data.dtypes

# %%
data_dategroupby = data.groupby('date')
days = list(data_dategroupby.groups.keys())
f"There are only {len(days)} days worth of data in the final dataframe after the NaNs are dropped"

# %%
# %%time
latf='/volstore/spacebridge/nasa_scp_byu/2020/sir_lats.txt'
lonf='/volstore/spacebridge/nasa_scp_byu/2020/sir_lons.txt'
with open(latf, 'r') as f:
  byulats=np.loadtxt(f, delimiter=',')
with open(lonf, 'r') as f:
  byulons=np.loadtxt(f, delimiter=',')
nasa_byu_tree = cKDTree(np.c_[byulons.ravel(), byulats.ravel()])
  
oi_conc = np.array([])
nasa_byu = np.array([])
# df = pd.DataFrame()
for day in days:
  logger.info(f"processing {day}")
  day=str(day)
  daydate=datetime.datetime.strptime(day, '%Y%m%d')
  prevdate=daydate - datetime.timedelta(1)
  yday=daydate.strftime('%j')
  ydaym1=prevdate.strftime('%j')
  y,m,d = day[:4], day[4:6], day[6:8]
  df = data_dategroupby.get_group(int(y+m+d))[['longitude', 'latitude']]
  logger.info(f'looking up {len(df)} locations')

  logger.info('Matching osisaf data')
  oic = xr.open_dataset(f'/volstore/spacebridge/iceconc_osisaf/{y}/{m}/ice_conc_sh_polstere-100_multi_{y+m+d}1200.nc')
  tree = cKDTree(np.c_[oic.lon.values.ravel(), oic.lat.values.ravel()])
  dd, ii = tree.query(df)
  oi_conc = np.append(oi_conc, oic.ice_conc.values.ravel()[ii])

  logger.info('Matching nasa byu data')
  rotf=f'/volstore/spacebridge/nasa_scp_byu/{y}/a_imgfiles/msfa-a-Ant{daydate.strftime("%y")}-{ydaym1}-{yday}.sir.img'
  if os.path.isfile(rotf):
    with open(rotf, 'rb') as f:
      temprot=np.fromfile(f, dtype=np.float32)
      rot=np.reshape(temprot, [1940,1940])
      # flip array vertically to correct orientation    
      rot = rot[::-1,:]
    dd, ii = nasa_byu_tree.query(df)
    byuday = rot.ravel()[ii]
  else:
    logger.info('Warning: file not found')
    byuday = np.array([np.nan]*len(df))
  nasa_byu = np.append(nasa_byu, byuday)

# %%
import pickle
with open('/home/stefan/GIT_REPOS/spacebridge/tempdata.pkl' ,'wb') as f:
  pickle.dump([oi_conc, nasa_byu],f) 

# %%
assert len(oi_conc)==len(data)
assert len(nasa_byu)==len(data)

# %%
data['oi_conc']=oi_conc
data['nasa_byu']=nasa_byu
data

# %%
data.to_csv('/home/stefan/GIT_REPOS/spacebridge/products/preprocessed_gnssr_update202330.csv', sep=',', header=True, index=False)
# !sudo mv /home/stefan/GIT_REPOS/spacebridge/products/preprocessed_gnssr_update202330.csv /volstore/spacebridge/gnssr_grzice/preprocessed_data/
# moved to /volstore/spacebridge afterwards
