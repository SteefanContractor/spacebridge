# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Add description here
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
import datetime
import glob
import xarray as xr
import sys

sys.path.append('/home/stefan/GIT_REPOS/spacebridge/pkgs')
from data_wrangling.gnssr import *

# %%
dates = []
d = datetime.date(2020,3,1)
while d <= datetime.date(2020,12,31):
    dates.append(datetime.datetime.strftime(d,'%Y-%m-%d')) #
    d += datetime.timedelta(days=1)

# %%
df = pd.DataFrame()
icetypefileprefix = '/volstore/spacebridge/iceage_Melsheimer/Icetypes/ECICE-IcetypesUncorrected-'
for date in dates:
    print(date)
    path = '/volstore/spacebridge/gnssr_grzice/data/'
    prefix = 'spire_gnss-r_L2_grzIce_v07.00_'
    files = glob.glob(path + prefix + date + "*")
    datadate = ""
    for i, file in enumerate(files):
        gnssr = xr.open_dataset(file)
        data, columns = extract_variables(gnssr, normalize=False)
        if len(data) > 0:
            dfnew = pd.DataFrame(data=data, columns=columns)
            if dfnew.date[0] != datadate:
                datadate = dfnew.date[0]
                icetypefile = icetypefileprefix + dfnew.date[0] + '.nc'
                print(f'Opening ice type file {icetypefile}')
                icetypeds = xr.open_dataset(icetypefile)
            dfnew = create_label_columns(dfnew, icetypeds)
            df = pd.concat([df,dfnew])
    [print(f'processed {i} files with date {date}') if len(files) > 0 else None]

print(f'total number of observations are {len(df)}')
df.to_csv('/home/stefan/GIT_REPOS/spacebridge/products/preprocessed_gnssr.csv', sep=',', header=True, index=False)
