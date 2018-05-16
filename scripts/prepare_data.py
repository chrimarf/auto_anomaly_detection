# coding: utf-8

# Preparing data to compare our method

import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_kddcup99
from sklearn.datasets.mldata import fetch_mldata

# Loading and editing datasets
# 
# The target variable contains the label of abnormality.
# 0 : Normal
# 1 : Anomaly

covtype = fetch_covtype()
SF = fetch_kddcup99(subset='SF')
http = fetch_kddcup99(subset='http')
shuttle = fetch_mldata('shuttle')

# We use the rules proposed in Learning hyperparameters for unsupervised anomaly detection.
# A. Thomas, S. Clémençon, V. Feuillard, A. Gramfort. Anomaly Detection Workshop, ICML 2016]

# For the Forest Cover dataset cover types 4 and 5 are considered abnormal when the cover type 2 is considered as normal

df_covtype = pd.DataFrame(covtype.data)
df_covtype['target'] = covtype.target
df_covtype = df_covtype.query('target in [2,4,5]')
df_covtype.target = df_covtype.target.replace(2, 0).replace(4, 1).replace(5, 1)

# For the sf and http dataset all the categories not flagged normal are considered abnormal

df_sf = pd.DataFrame(SF.data)
df_sf['target'] = SF.target
df_sf.target[df_sf.target != 'normal.'] = 1
df_sf.target = df_sf.target.replace('normal.', 0)

df_http = pd.DataFrame(http.data)
df_http['target'] = http.target
df_http.target[df_http.target != 'normal.'] = 1
df_http.target = df_http.target.replace('normal.', 0)

df_shuttle = pd.DataFrame(shuttle.data)
df_shuttle['target'] = shuttle.target
df_shuttle = df_shuttle.query('target != 4')
df_shuttle.target = df_shuttle.target.replace(1, 0)
df_shuttle.target[df_shuttle.target != 0] = 1

# This dataset deals with fraudulent activities on credit cards and has been released with
# Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
# Calibrating Probability with Undersampling for Unbalanced Classification.
# In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

df_creditcard = pd.read_csv('data/creditcard.csv')
df_creditcard = df_creditcard.rename(columns={"Class": "target"})

alldfs = [var for var in dir() if
          (isinstance(eval(var), pd.core.frame.DataFrame)) and (var != 'stats_df') and (var[:2] == 'df')]
stats_df = pd.DataFrame(columns=['name', 'ncol', 'nrow', 'anomaly_percentage'])
stats_df['name'] = alldfs
stats_df['ncol'] = [len(locals()[df].columns) for df in alldfs]
stats_df['nrow'] = [len(locals()[df].index) for df in alldfs]
stats_df['anomaly_percentage'] = [len((locals()[df]).query('target == 1').index) for df in alldfs]
stats_df['anomaly_percentage'] /= stats_df['nrow']

print 'Attributes of the datasets used'
print '--------------------------------'
print stats_df

### Save Data

[locals()[df].to_pickle('data/pickle_datasets/{name}.pkl'.format(name=df)) for df in alldfs]
