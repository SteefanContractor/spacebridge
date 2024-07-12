import pandas as pd
import pickle
from datetime import datetime
import umap

if __name__=='__main__':
    # load preprocessed data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    lab_names =  ['YI_conc','FYI_conc','MYI_conc','water_conc']
    # load data
    data_path = "../data/preprocessed_gnssr_update202330_clean/"
    Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
    features = Xres.sample(frac=0.12, random_state=42)
    labels = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0)
    labels = labels.loc[features.index]
    print(labels.value_counts())

    MIN_DIST = 0.5
    N_NEIGHBORS = 75
    N_COMPONENTS = 5
    start = datetime.now()
    trans_supervised = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=N_COMPONENTS, random_state=42).fit(features, y=pd.factorize(labels['0'])[0])
    end = datetime.now()
    print(f"Time elapsed: {end-start}")

    # save the transformation
    timestamp = datetime.now().strftime("%Y%m%d:%H%M%S")
    print(f"Saving the transformation at {timestamp}")
    with open(f'../products/models/umap_transformation/umap_1200Kresampledfeats_mindist{MIN_DIST}_neighbors{N_NEIGHBORS}_numcomp{N_COMPONENTS}_{timestamp}.pkl', 'wb') as file:
        pickle.dump(trans_supervised, file)
