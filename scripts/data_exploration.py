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
# Makes some plots of the preprocessed data as simple sanity checks
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
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
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# %%
import sys
import gc
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# %% [markdown]
# # Labels exploration
#
# ## OISSAF ice concentration labels

# %%
logger.info('Loading OSISAF data: ' + data_path+'preprocessed_gnssr_update202330.csv')

# %%
# load preprocessed data
label = ["oi"]
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))] + [data_lab_name[l] for l in label]
dtype = dict({c: 'float64' for c in orig_feats if c not in list(data_lab_name.values())+['date']}, **{c: 'float32' for c in orig_feats if c in data_lab_name.values()})
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[orig_feats] # reorder columns
# data = data.astype({orig_feats[-1]: 'float32'}) # we do not need greater than float32 precision for the label column
data.dropna(inplace=True)
orig_feats = data.keys()[2:-1] # original features are the features used for training; so no labels or data/time
data.reset_index(inplace= True, drop=True)
data

# %%
data.dtypes

# %%
orig_feats

# %%
plt.subplots(1,1,subplot_kw=dict(projection=ccrs.Orthographic(0,-90)), figsize=(15,18))
plt.scatter(data.longitude, data.latitude, transform=ccrs.PlateCarree(), c='k', alpha=0.01, s=1)

# %%
datentime = data[['date','time']]#.drop(data.sample(frac=0.9,random_state=202206).index)
winterdatetime = datentime.query('20200501 <= date <= 20200930')
summerdatetime = datentime.query('date < 20200401 or date >= 20201101')
AprOctdatetime = datentime.drop(pd.concat([winterdatetime, summerdatetime]).index)

# %%
winterdata=data.loc[winterdatetime.index]
summerdata=data.loc[summerdatetime.index]
shoulderdata=data.loc[AprOctdatetime.index]

# %%
logger.info('Plotting histogram')

# %%
plt.figure(figsize=(15,8))
lab='oi_conc'
plt.hist([data[lab], winterdata[lab], summerdata[lab], shoulderdata[lab]], bins=40, density=True, 
         label=[f'All 2020; N={len(data)}', f'Winter (MJJA); N={len(winterdata)}', f'Summer (JFM+ND); N={len(summerdata)}', f'Shoulder (A+SO); N={len(summerdata)}'])
plt.xlabel('ice concentration (%)')
plt.legend()
plt.title('OSISAF sea ice concentration')

# %%
logger.info('Plotting coordinates')

# %%
seas = ['All 2020', 'Winter (MJJA)', 'Summer (JFM+ND)', 'Shoulder (A+SO)']
presencelab = ['no ice', 'ice']
lab='oi_conc'
nonzeroall = {plab: data[data[lab] > 0] if lab=='ice' else data[data[lab]==0] for plab in presencelab}
nonzerowinter = {plab: winterdata[winterdata[lab] > 0] if lab=='ice' else winterdata[winterdata[lab]==0] for plab in presencelab}
nonzerosummer = {plab: summerdata[summerdata[lab] > 0] if lab=='ice' else summerdata[summerdata[lab]==0] for plab in presencelab}
nonzeroshoulder = {plab: shoulderdata[shoulderdata[lab] > 0] if lab=='ice' else shoulderdata[shoulderdata[lab]==0] for plab in presencelab}
nonzerodata = [nonzeroall, nonzerowinter, nonzerosummer, nonzeroshoulder]

# %%
fig, axs = plt.subplots(2,2,subplot_kw=dict(projection=ccrs.Orthographic(0,-90)), figsize=(15,18))
for i,ax in enumerate(axs.flatten()):
  for plab in presencelab:
    if plab == 'no ice':
      # pass
      ax.scatter(nonzerodata[i][plab]['longitude'], nonzerodata[i][plab]['latitude'], transform=ccrs.PlateCarree(), c='k', alpha=0.01, s=1)
    else:
      # pass
      ax.scatter(nonzerodata[i][plab]['longitude'], nonzerodata[i][plab]['latitude'], transform=ccrs.PlateCarree(), c='r', alpha=0.01, s=1)
  ax.coastlines(resolution='50m')
  ax.set_title(seas[i])
plt.tight_layout()

# %%
# memory management
del data, winterdata, summerdata, shoulderdata
gc.collect()

# %%
gc.collect()

# %% [markdown]
# ## University of Bremen MultiYear Ice Concentration

# %%
logger.info('Loading Bremen Multi Year Data')

# %%
# load preprocessed data
label = ['yi',"myi",'fyi']
data_lab_name = dict(oi='oi_conc', # these are the column names in the saved preprocessed csv
                     yi='YI_conc',
                     myi='MYI_conc',
                     fyi='FYI_conc')
# read the header to get the column names
orig_feats = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv',nrows=1).columns.tolist()
orig_feats = [orig_feats[i] for i in list(range(18))]+[data_lab_name[l] for l in label]#[orig_feats.index(data_lab_name[label])]]
dtype = dict({c: 'float64' for c in orig_feats if c not in list(data_lab_name.values())+['date']}, **{c: 'float32' for c in orig_feats if c in data_lab_name.values()})
# # read values
data = pd.read_csv(data_path+'preprocessed_gnssr_update202330.csv', usecols=orig_feats,dtype=dtype,parse_dates=['date'],low_memory=True)
data = data[orig_feats] # reorder columns
data.dropna(inplace=True)
# orig_feats = data.keys()[2:-1] # original features are the features used for training; so no labels or data/time
data.reset_index(inplace= True, drop=True)
data

# %%
data['totm_conc'] = data.YI_conc + data.FYI_conc + data.MYI_conc

# %%
data.dtypes

# %%
datentime = data[['date','time']]#.drop(data.sample(frac=0.9,random_state=202206).index)
winterdatetime = datentime.query('20200501 <= date <= 20200930')
summerdatetime = datentime.query('date < 20200401 or date >= 20201101')
AprOctdatetime = datentime.drop(pd.concat([winterdatetime, summerdatetime]).index)

# %%
winterdata=data.loc[winterdatetime.index]
summerdata=data.loc[summerdatetime.index]
shoulderdata=data.loc[AprOctdatetime.index]

# %%
logger.info('Plotting histogram')

# %%
label=['totm_conc','YI_conc','MYI_conc','FYI_conc']
fig, axs = plt.subplots(4,1,figsize=(8,9),sharex=True)
fig.suptitle('U. Brem. Multiage Sea Ice Concentration', verticalalignment='center')
for ax, lab in zip(axs, label):
  ax.hist([data.loc[data[lab]>0,lab], winterdata.loc[winterdata[lab]>0,lab], summerdata.loc[summerdata[lab]>0,lab], shoulderdata.loc[shoulderdata[lab]>0,lab]], 
          bins=40, density=True, histtype='bar',
           label=[f'All 2020; N={len(data)}', f'Winter (MJJA); N={len(winterdata)}', f'Summer (JFM+ND); N={len(summerdata)}', f'Shoulder (A+SO); N={len(summerdata)}'])
  ax.set_xlabel('Non zero '+lab+' ice concentration (%)')
  ax.legend() if lab in ['MYI_conc'] else None
plt.tight_layout


# %% [markdown]
# ### Plot coordinates of GNSS-R 

# %%
logger.info('Plotting coordinates')

# %%
seas = ['All 2020', 'Winter (MJJA)', 'Summer (JFM+ND)', 'Shoulder (A+SO)']
# label = [label[i] for i in [-1,0,1,2]]
nonzeroall = {lab: data[data[lab] > 0] if lab!='totm_conc' else data[data[lab]==0] for lab in label}
nonzerowinter = {lab: winterdata[winterdata[lab] > 0] if lab!='totm_conc' else winterdata[winterdata[lab]==0] for lab in label}
nonzerosummer = {lab: summerdata[summerdata[lab] > 0] if lab!='totm_conc' else summerdata[summerdata[lab]==0] for lab in label}
nonzeroshoulder = {lab: shoulderdata[shoulderdata[lab] > 0] if lab!='totm_conc' else shoulderdata[shoulderdata[lab]==0] for lab in label}
nonzerodata = [nonzeroall, nonzerowinter, nonzerosummer, nonzeroshoulder]

# %%
fig, axs = plt.subplots(4,4,subplot_kw=dict(projection=ccrs.Orthographic(0,-90)), figsize=(15,18))
for si, s in enumerate(seas):
  for li, lab in enumerate(label):
    # if lab == 'totm_conc':
    axs[si,li].scatter(nonzerodata[si][lab]['longitude'], nonzerodata[si][lab]['latitude'], transform=ccrs.PlateCarree(), c='k', alpha=0.01, s=1)
    if li==0:
      axs[si,li].set_ylabel(seas[si], size='large')
    if si==0:
      axs[si,li].set_title(label[li])
plt.tight_layout()

# %%
