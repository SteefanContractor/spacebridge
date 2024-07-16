# %%
import pandas as pd
import pickle 
import datetime
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
# %%
data_path = "../data/preprocessed_gnssr_update202330_clean/"
lab_names = pd.read_csv(data_path + "labels.csv", nrows=1, index_col=0).columns.tolist()
dtype = {label: 'float32' for label in lab_names}
labels = pd.read_csv(data_path + "labels.csv", index_col=0, dtype=dtype)
# subset to sure labels
labels = labels[(labels.YI_conc > 90.) | (labels.FYI_conc > 99.9) | (labels.MYI_conc > 99.) | (labels.water_conc == 100.)]
# drop ice_conc column
labels.drop('ice_conc', axis=1, inplace=True)
lab_names = [lab for lab in lab_names if lab != 'ice_conc']
label_encoded = labels.idxmax(axis=1)
# count the number of each label
label_encoded.value_counts()
# put 10% aside for testing
test_label = label_encoded.sample(frac=0.1, random_state=42)
display(test_label.value_counts())
# %%
feat_names = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, nrows=1).columns.tolist()
dtype = {feat: 'float64' for feat in feat_names}
train_features = pd.read_csv(data_path + "transformed_scaled_feats.csv", index_col=0, dtype=dtype)
train_features = train_features.loc[label_encoded.index]
train_features = train_features.drop(test_label.index, axis=0)
train_features = train_features.sample(frac=0.05, random_state=42)
train_labels = label_encoded.loc[train_features.index]
print(train_labels.value_counts())
train_label_int = train_labels.astype('category').cat.codes
# %%
# data_path = "../data/preprocessed_gnssr_update202330_clean/"
# Xres = pd.read_csv(data_path + "resampled/SMOTEENN_feats.csv", index_col=0)
# yres = pd.read_csv(data_path + "resampled/SMOTEENN_labels.csv", index_col=0).iloc[:, 0]
# train_features = Xres.sample(frac=0.012, random_state=42)
# train_label = yres.loc[train_features.index]
# print(train_label.value_counts())
# train_label_int = train_label.astype('category').cat.codes

# load umap transformation
# umap = pickle.load(open('../products/models/umap_transformation/umap_1200Kresampledfeats_mindist0.5_neighbors75_numcomp5_20240514:111207.pkl', "rb"))
# train_features = umap.transform(train_features)
# %%
scoring = ['accuracy', 'r2', 'matthews_corrcoef', 'roc_auc_ovo_weighted', 'f1_weighted']
model_names = ['LogisticRegression', 'DecisionTreeClassifier', 'KNeighborsClassifier',
  'LinearDiscriminantAnalysis', 'GaussianNB', 'RandomForestClassifier',
  'GradientBoostingClassifier', 'AdaBoostClassifier', 'HistGradientBoostingClassifier',
  'MLPClassifier', 'SVC']
models = [LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(),
  LinearDiscriminantAnalysis(), GaussianNB(), RandomForestClassifier(),
  GradientBoostingClassifier(), AdaBoostClassifier(), HistGradientBoostingClassifier(),
  MLPClassifier(), SVC()]
results_dict = {}
for name, model in zip(model_names, models):
    print('Running ' + name)
    results = model_selection.cross_validate(model, train_features, train_label_int, 
                                             cv=5, n_jobs=8, verbose=1,
                                             scoring=scoring)
    results_dict[name] = results
# %%
# save results_dict
timestamp = datetime.datetime.now().strftime("%Y%m%d:%H%M%S")
with open(f'../products/results/compare_skmodels_results_dict_{timestamp}.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
pickle.dump(results_dict, open('results_dict.pkl', 'wb'))
# %%
for score in results_dict['LogisticRegression'].keys():
  fig = plt.figure()
  fig.suptitle(score)
  ax = fig.add_subplot(111)
  plt.boxplot(np.array([res[score] for res in results_dict.values()]).T)
  ax.set_xticklabels(model_names, rotation=45, ha='right')
  plt.show()
# %%
