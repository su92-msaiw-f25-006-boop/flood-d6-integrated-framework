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
y = d.serious.values; pct = d.pct.values; g = d.key.values
def mk(nm, seed=42):
    if nm == 'LR': return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
    if nm == 'RF': return RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=2)
    if nm == 'XGB': return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, eval_metric='logloss', scale_pos_weight=(y == 0).sum()/(y == 1).sum(), random_state=seed, n_jobs=2)
lg = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6)))
def base_feats(Xa, ya, pa_, Xb, ga, seed=42):
    """return base-model probabilities for Xb (LR, RF, XGB, Linear-regression %) trained on (Xa)"""
    out = []
    for nm in ('LR', 'RF', 'XGB'): m = mk(nm, seed).fit(Xa, ya); out.append(m.predict_proba(Xb)[:, 1])
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(Xa, np.clip(pa_, 0, 100)); out.append(np.clip(lin.predict(Xb), 0, 100)/100)
    return np.column_stack(out)
def stack_fit_predict(Xa, ya, pa_, ga, Xb, seed=42):
    inner = GroupKFold(4); O = np.zeros((len(Xa), 4))
    for i, j in inner.split(Xa, ya, ga): O[j] = base_feats(Xa[i], ya[i], pa_[i], Xa[j], ga[i], seed)
    meta = LogisticRegression(C=1., class_weight='balanced', max_iter=2000).fit(np.column_stack([lg(O[:, :3]), O[:, 3:]]), ya)
    B = base_feats(Xa, ya, pa_, Xb, ga, seed); return meta.predict_proba(np.column_stack([lg(B[:, :3]), B[:, 3:]]))[:, 1], B, meta.coef_[0].round(2)
def met(yt, p, th=.5):
    pr = (p >= th).astype(int); return dict(auc=roc_auc_score(yt, p), prauc=average_precision_score(yt, p), f1=f1_score(yt, pr), prec=precision_score(yt, pr, zero_division=0), rec=recall_score(yt, pr))
def boot(yt, p, grp, B=1000, th=.5, seed=0):
    rng = np.random.default_rng(seed); u = np.unique(grp); f, a = [], []
    idx = {k: np.where(grp == k)[0] for k in u}
    for _ in range(B):
        ix = np.concatenate([idx[k] for k in rng.choice(u, len(u))]);
        if yt[ix].min() == yt[ix].max(): continue
        f.append(f1_score(yt[ix], p[ix] >= th)); a.append(roc_auc_score(yt[ix], p[ix]))
    return [np.percentile(f, [2.5, 97.5]).round(3).tolist(), np.percentile(a, [2.5, 97.5]).round(3).tolist()]
R = {}; OOF = {}
for sname, cols in SETS.items():
    X = d[cols].values.astype(float); S = {nm: np.zeros(len(d)) for nm in ('LR', 'RF', 'XGB', 'LinReg', 'STACK')}; coefs = []
    for i, j in GroupKFold(5).split(X, y, g):
        pst, B, cf = stack_fit_predict(X[i], y[i], pct[i], g[i], X[j]); coefs.append(cf)
        for k, nm in enumerate(('LR', 'RF', 'XGB', 'LinReg')): S[nm][j] = B[:, k]
        S['STACK'][j] = pst
    for nm, p in S.items(): R[f'{nm} | {sname}'] = met(y, p) | dict(ci=boot(y, p, g)); OOF[f'{nm}|{sname}'] = p
    R[f'LinReg R2 | {sname}'] = r2_score(pct, S['LinReg']*100)
    print(sname, {k: round(v['auc'], 3) for k, v in R.items() if k.endswith(sname) and isinstance(v, dict)}, 'F1 stack', round(R[f'STACK | {sname}']['f1'], 3), flush=True)
# 2025 hold-out and per-province, fused set
cols = SETS['D5 + D4']; X = d[cols].values.astype(float); tr = d.year < 2025; te = d.year == 2025
pst, B, cf = stack_fit_predict(X[tr], y[tr], pct[tr], g[tr], X[te]); yt = y[te.values]
R['holdout2025'] = {'n': int(te.sum()), 'floods': int(yt.sum()), 'STACK': met(yt, pst), 'LR': met(yt, B[:, 0]), 'RF': met(yt, B[:, 1]), 'XGB': met(yt, B[:, 2]), 'coef': cf.tolist()}
pr = {}
for pv in d.province.unique():
    m = (d.province == pv).values
    if y[m].sum() >= 2 and (y[m] == 0).sum() >= 2: pr[pv] = {'n': int(m.sum()), 'floods': int(y[m].sum()), 'auc_stack': roc_auc_score(y[m], OOF['STACK|D5 + D4'][m]), 'auc_lr_d5': roc_auc_score(y[m], OOF['LR|D5 (terrain+rain)'][m])}
    else: pr[pv] = {'n': int(m.sum()), 'floods': int(y[m].sum()), 'note': 'too few flood/non-flood rows for AUC'}
R['province'] = pr; d.assign(**{k.replace('|', '_').replace(' ', ''): v for k, v in OOF.items() if k.startswith('STACK')})[['key', 'year', 'province', 'serious', 'pct']].assign(stack=OOF['STACK|D5 + D4'], lr=OOF['LR|D5 (terrain+rain)']).to_csv('oof_a.csv', index=False)
json.dump(R, open('results_a.json', 'w'), indent=1, default=float); print('holdout2025', {k: (round(v['auc'], 3) if isinstance(v, dict) else v) for k, v in R['holdout2025'].items()}); print(pr); print('done')
