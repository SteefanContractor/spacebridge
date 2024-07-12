# %%
import pandas as pd
import pickle
from datetime import datetime
import umap
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
# %%
# load preprocessed data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
lab_names =  ['YI_conc','FYI_conc','MYI_conc','water_conc']
# load data
data_path = "../data/preprocessed_gnssr_update202330_clean/"
Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
features = Xres.sample(frac=0.02, random_state=42)
labels = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0)
labels = labels.loc[features.index]
# dtype = dtype = {label: 'float32' for label in lab_names}
# labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype=dtype, usecols=['index']+lab_names)
# # subset to sure labels
# labels = labels[(labels.YI_conc > 90.) | (labels.FYI_conc > 99.9) | (labels.MYI_conc > 99.) | (labels.water_conc == 100.)]
# labels = labels.idxmax(axis=1)
labels.value_counts()
# %%
%%time
trans = umap.UMAP(n_neighbors=15, n_components=3, random_state=42).fit(features) # default n_neighbors=15
# %%
# %%time
# umap3d_feats = trans.transform(features)
umap3d_feats = pd.DataFrame(umap3d_feats, index=features.index, columns=['x', 'y', 'z'])
# visualize randomly sampled 10000 points
# sampled_indices = np.random.choice(umap3d_feats.shape[0], size=10000, replace=False)
sampled_points = umap3d_feats.sample(frac=0.05, random_state=42)
# create an interactive 3D scatter plot using plotly
fig = px.scatter_3d(umap3d_feats, color=labels['0'], x='x', y='y', z='z', 
                    opacity=0.7, size_max=2,
                    width=1200, height=800,)

# Adjust the marker size properties
fig.update_traces(marker=dict(size=2,  # Specify the desired point size
                            #   opacity=0.7,  # Adjust point opacity
                            #   line=dict(width=2, color='DarkSlateGrey')
                              ))  # Customize marker line

# show the plot
fig.show()
# visualize_3d(sampled_points, labels)

# visualize_3d(umap3d_feats, labels)
# %%
%%time
MIN_DIST = 0.5
N_NEIGHBORS = 60
trans_supervised = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=42).fit(features, y=pd.factorize(labels['0'])[0])
# %%
print(f'min_dist: {MIN_DIST}, n_neighbors: {N_NEIGHBORS}')
umap3d_feats = trans_supervised.transform(features)
umap3d_feats = pd.DataFrame(umap3d_feats, index=features.index, columns=['x', 'y', 'z'])
# visualize randomly sampled 10000 points
# sampled_indices = np.random.choice(umap3d_feats.shape[0], size=10000, replace=False)
sampled_points = umap3d_feats.sample(frac=0.05, random_state=42)
# create an interactive 3D scatter plot using plotly
fig = px.scatter_3d(umap3d_feats, color=labels['0'], x='x', y='y', z='z', 
                    opacity=0.7, size_max=2,
                    width=1000, height=800,)

# Adjust the marker size properties
fig.update_traces(marker=dict(size=2,  # Specify the desired point size
                            #   opacity=0.7,  # Adjust point opacity
                            #   line=dict(width=2, color='DarkSlateGrey')
                              ))  # Customize marker line

# show the plot
fig.show()
# %%
# save the transformation
timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
print(f"Saving the transformation at {timestamp}")
with open(f'../products/models/umap_transformation/umap_200Kresampledfeats_mindist{MIN_DIST}_neighbors{N_NEIGHBORS}_{timestamp}.pkl', 'wb') as file:
    pickle.dump(trans_supervised, file)

# %%
with open('../products/models/umap_transformation/umap_200Kresampledfeats_mindist0.5_neighbors60_20240513:230240.pkl', 'rb') as file:
    test = pickle.load(file)
# %%
