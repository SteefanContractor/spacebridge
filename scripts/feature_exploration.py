#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.decomposition import PCA
# %%
# load preprocessed data
data_path = '../data/'
label = ['yi','myi','fyi']
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
# drop -999.0 values from reflectivity1 and reflectivity2 columns
data = data[data.reflectivity1!=-999.0]
data = data[data.reflectivity2!=-999.0]
data.dropna(inplace=True)
orig_feats = data.keys()[2:-5] # original features are the features used for training; so no labels or data/time
labels = [data_lab_name[l] for l in label]
if len(labels)>1:
    data['tot_conc'] = data[labels].sum(axis=1)
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
data.dropna(inplace=True)
# %%
# drop rows with excess_phase_noise2 < -5 (only 35 rows)
data = data[data.excess_phase_noise2>-5.]
# %%
# min-max scale original features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[orig_feats] = scaler.fit_transform(data[orig_feats])
# %%
# plot histograms of feature columns
data[orig_feats].hist(figsize=(20,20),bins=50)
# %%
# plot heatmap of feature correlations
plt.figure(figsize=(20,20))
sns.heatmap(data[orig_feats].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# %%
# heatmap of feature correlations with labels
plt.figure(figsize=(20,20))
sns.heatmap(data[list(orig_feats)+labels].corr(), annot=True, fmt='.2f', cmap='coolwarm')
# %%
# filter tot_conc == 0 indicating water and plot histogram of original features
data[data.tot_conc==0.][orig_feats].hist(figsize=(20,20),bins=50)
# %%
# filter tot_conc > 99. indicating ice and plot histogram of original features
data[data.tot_conc>99.][orig_feats].hist(figsize=(20,20),bins=50)
# %%
# investigate rows with total concentration above 99% and 99.9%
def icetype_dist(df):
    print(f'total number of rows: {len(df)}')
    print(f'number of YI_conc rows: {len(df[df.YI_conc>0.])}')
    print(f'number of MYI_conc rows: {len(df[df.MYI_conc>0.])}')
    print(f'number of FYI_conc rows: {len(df[df.FYI_conc>0.])}')

print('total concentration > 99%')
icetype_dist(data[data.tot_conc > 99.])
print('total concentration > 99.9%')
icetype_dist(data[data.tot_conc > 99.9])          
# %%
# subset data to only include water and ice with equal fractions of each
waterice = data[(data.tot_conc==0.) | (data.tot_conc>99.)]
waterice['waterice_label'] = [0 if tc==0. else 1 for tc in waterice.tot_conc]
# save waterice dataframe to file
waterice.to_csv(data_path+'waterice.csv',index=False)
waterice = waterice.groupby('waterice_label').sample(frac=0.2,random_state=42)
# %%
# project data rows onto 2D space using PCA
# PCA features
pca_feats = ['reflectivity1','snr_reflected1','power_reflected1','phase_noise1','excess_phase_noise1',]
pca = PCA(n_components=3)
pca_results = pca.fit_transform(waterice[pca_feats])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# %%
#Plot PCA results
# create a DataFrame with the 10% of PCA results and ice type labels
# randomly sample 10% of the indices of pca_results
idx = np.random.choice(pca_results.shape[0], int(pca_results.shape[0]*0.1), replace=False)
df = pd.DataFrame({'PC1': pca_results[idx,0], 'PC2': pca_results[idx,1], 'PC3': pca_results[idx,2], 'water_or_ice': waterice['waterice_label'].iloc[idx]})

# create an interactive 3D scatter plot using plotly
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='water_or_ice')

# Adjust the marker size properties
fig.update_traces(marker=dict(size=2,  # Specify the desired point size
                            #   opacity=0.7,  # Adjust point opacity
                            #   line=dict(width=2, color='DarkSlateGrey')
                              ))  # Customize marker line

# show the plot
fig.show()
fig.write_html("../products/notebooks/feature_exploration/First three principal components of features coloured by water-ice label.html")
# %%
distinct_ice_types = data[(data.YI_conc>90.) | (data.MYI_conc>99.) | (data.FYI_conc>99.9)]
distinct_ice_types['ice_type'] = ['YI' if yi>90. else 'MYI' if myi>99. else 'FYI' for yi,myi in zip(distinct_ice_types.YI_conc,distinct_ice_types.MYI_conc)]
# save distinct_ice_types dataframe to file
distinct_ice_types.to_csv(data_path+'distinct_ice_types.csv',index=False)
pca_feats = ['reflectivity1','snr_reflected1','power_reflected1','phase_noise1','excess_phase_noise1',]
# %%
# investigate what percentiles YI>90., MYI>99., and FYI>99.9 refer to
print('total number of rows in distinct_ice_types: ', len(distinct_ice_types))
print(f'YI_conc > 90. results in {len(data[data.YI_conc>90.])} rows')
print(f'which corresponds to top {len(data[data.YI_conc>90.])/len(distinct_ice_types)*100:.2f}% of YI_conc values in distinct_ice_types')
print(f'MYI_conc > 99. results in {len(data[data.MYI_conc>99.])} rows')
print(f'which corresponds to top {len(data[data.MYI_conc>99.])/len(distinct_ice_types)*100:.2f}% of MYI_conc values in distinct_ice_types')
print(f'FYI_conc > 99.9 results in {len(data[data.FYI_conc>99.9])} rows')
print(f'which corresponds to top {len(data[data.FYI_conc>99.9])/len(distinct_ice_types)*100:.2f}% of FYI_conc values in distinct_ice_types')
print(f'FYI_conc > 99.9 results in {len(data[data.FYI_conc>99.99])} rows')
print(f'which corresponds to top {len(data[data.FYI_conc>99.99])/len(distinct_ice_types)*100:.2f}% of FYI_conc values in distinct_ice_types')

# %%
# project data rows onto 3D space using PCA
# PCA features
pca_feats = ['reflectivity1','snr_reflected1','power_reflected1','phase_noise1','excess_phase_noise1',]
pca = PCA(n_components=3)
pca_results = pca.fit_transform(distinct_ice_types[pca_feats])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# %%
# create a DataFrame with the PCA results and ice type labels
df = pd.DataFrame({'PC1': pca_results[:,0], 'PC2': pca_results[:,1], 'PC3': pca_results[:,2], 'ice_type': distinct_ice_types['ice_type']})

# create an interactive 3D scatter plot using plotly
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='ice_type')

# Adjust the marker size properties
fig.update_traces(marker=dict(size=1,  # Specify the desired point size
                            #   opacity=0.7,  # Adjust point opacity
                            #   line=dict(width=2, color='DarkSlateGrey')
                              ))  # Customize marker line
fig.update_layout(legend=dict(x=0.8))
# show the plot
fig.show()
fig.write_html("../products/notebooks/feature_exploration/First three principal components of features coloured by ice type labels.html")
# %%
# Check if there is more separation between ice types seasonally
# create seasonal dataframe of distinct ice types
# get seasonal indices by querying the date + time columns
datentime = distinct_ice_types[['date','time']]#.drop(data.sample(frac=0.9,random_state=202206).index)
winterdatetime = datentime.query('20200501 <= date <= 20200930')
summerdatetime = datentime.query('date < 20200401 or date >= 20201101')
AprOctdatetime = datentime.drop(pd.concat([winterdatetime, summerdatetime]).index)
# use indices to filter data
wintericetypes=distinct_ice_types.loc[winterdatetime.index]
summericetypes=distinct_ice_types.loc[summerdatetime.index]
shouldericetypes=distinct_ice_types.loc[AprOctdatetime.index]
# assert that the length of the seasonal dataframes is equal to the length of distinct_ice_types
assert len(wintericetypes)+len(summericetypes)+len(shouldericetypes)==len(distinct_ice_types)
# %%
def plot_pca(df, feats, n_components=3, label=None):
    """
    Plot PCA results in 2D or 3D space using plotly.
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing PCA results and labels.
    feats : list
        List of feature names.
    n_components : int, optional
        Number of PCA components to plot (2 or 3). The default is 3.
    label : str, optional
        Name of label column. The default is None.
    Returns
    -------
    df_pca : pandas DataFrame
        DataFrame containing PCA results and labels.
    """ 
    assert n_components in [2,3], 'n_components must be 2 or 3'
    # perform PCA on the DataFrame
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(df[feats])

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    # create a DataFrame with the PCA results and color labels
    columns = [f'PC{i+1}' for i in range(n_components)]
    if label is not None:
        df_pca = pd.DataFrame({col: pca_results[:,i] for i, col in enumerate(columns)})
        df_pca['label'] = df[label].reset_index(drop=True)
    else:
        df_pca = pd.DataFrame({col: pca_results[:,i] for i, col in enumerate(columns)})

    if n_components==2:
      # create an interactive 2D scatter plot using plotly
      fig = px.scatter(df_pca, x='PC1', y='PC2', color='label')
    elif n_components==3:
      # create an interactive 3D scatter plot using plotly
      fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='label')
      # Adjust the marker size properties
      fig.update_traces(marker=dict(size=2,  # Specify the desired point size
                                    ))  # Customize marker line
    # show the plot
    fig.show()

    return df_pca
# %%
# summer PCA
summer_pca = plot_pca(summericetypes, pca_feats, n_components=3, label='ice_type')
# winter PCA
winter_pca = plot_pca(wintericetypes, pca_feats, n_components=3, label='ice_type')
# shoulder PCA
shoulder_pca = plot_pca(shouldericetypes, pca_feats, n_components=3, label='ice_type')
# %%
