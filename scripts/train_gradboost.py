# %%
# import packages
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier
# %%
# read in the data
data = pd.read_csv('../data/distinct_ice_types.csv')
# drop the first two columns and the last seven columns except the very last column
data.drop(data.columns[:2].tolist()+data.columns[-7:-1].tolist(), axis=1, inplace=True)
# convert label to integer
label = 'ice_type'
data[label] = data[label].astype('category').cat.codes
# split into train/val at a random 85:15 ratio
train = data.sample(frac=0.85,random_state=42)
val = data.drop(train.index)
# %%
def build_clf(classifier, params):
    # use above as kwargs for lgbmclassifier
    clf = classifier(**params)
    # fit the model
    clf.fit(train.drop(label, axis=1), train[label])
    # train/val score
    print(f'Train score: {clf.score(train.drop(label, axis=1), train[label])}')
    print(f'Validation score: {clf.score(val.drop(label, axis=1), val[label])}')
    return clf
# %%
# set lgbm parameters
params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 3,
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
}

for nl in [31, 63, 127, 255, 511, 1023]:
    params['num_leaves'] = nl
    print(f'num_leaves: {nl}')
    build_clf(LGBMClassifier, params)
# %%
# set xgb parameters
params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'booster': 'gbtree',
    'learning_rate': 0.05,
    'max_depth': 6,
}

for md in [6, 7, 8, 9, 10, 11]:
    params['max_depth'] = md
    print(f'max_depth: {md}')
    build_clf(XGBClassifier, params)