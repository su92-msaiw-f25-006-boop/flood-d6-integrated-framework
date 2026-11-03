import os, sys, json, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import load_all
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, r2_score
from xgboost import XGBClassifier
D = load_all(); d = D['d5'].copy().reset_index(drop=True)
# D4: provincial rainfall anomaly of the same year (annual total vs 1981-2010 mean)
p4 = D['d4'].copy(); p4['dt'] = pd.to_datetime(p4.Date, format='%d-%m-%y', errors='coerce'); p4['yr'] = p4.dt.dt.year; p4['rain'] = p4['Rainfall - (MM)']
pa = p4.groupby(['Province', 'yr']).rain.sum().reset_index(); base = pa[(pa.yr >= 1981) & (pa.yr <= 2010)].groupby('Province').rain.mean()
pa['anom'] = pa.apply(lambda r: r.rain/base.get(r.Province, np.nan), axis=1)
pm = {'Khyber Pakhtunkhwa': 'KP', 'Balochistan': 'Balochistan', 'Punjab': 'Punjab', 'Sindh': 'Sindh', 'Islamabad Capital Territory': 'Islamabad'}
print(sorted(d.province.unique()), sorted(pa.Province.unique()))
d['prov4'] = d.province.map(lambda s: next((k for k in pa.Province.unique() if k.lower()[:4] == str(s).lower()[:4]), None))
d = d.merge(pa.rename(columns={'Province': 'prov4', 'yr': 'year', 'rain': 'prov_rain', 'anom': 'prov_anom'}), on=['prov4', 'year'], how='left')
d['prov_anom'] = d.prov_anom.fillna(d.prov_anom.median()); d['prov_rain'] = d.prov_rain.fillna(d.prov_rain.median())
T = ['elevation', 'slope', 'hand', 'log_flow_acc', 'cropland_frac', 'built_frac', 'flat_frac', 'elev_sd']; Rn = ['rain_total', 'rain_maxday']; P4 = ['prov_rain', 'prov_anom']
SETS = {'D5 terrain only': T, 'D5 rain only': Rn, 'D5 (terrain+rain)': T+Rn, 'D5 + D4': T+Rn+P4}

T = ['elevation', 'slope', 'hand', 'log_flow_acc', 'cropland_frac', 'built_frac', 'flat_frac', 'elev_sd']; Rn = ['rain_total', 'rain_maxday']; P4 = ['prov_rain', 'prov_anom']
