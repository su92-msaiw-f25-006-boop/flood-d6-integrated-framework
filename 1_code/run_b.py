import pandas as pd, numpy as np, warnings, json, sys; warnings.filterwarnings('ignore')
import torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
torch.set_num_threads(2)
df = pd.read_csv('fused_dadu_daily.csv', parse_dates=['date']).sort_values('date').reset_index(drop=True)
weather = ['precipitation', 'precip_3day_avg', 'precip_7day_avg', 'temperature', 'temp_3day_avg', 'humidity', 'wind_speed', 'pressure', 'soil_moisture', 'soil_3day_avg', 'evaporation', 'month', 'day_of_year', 'is_monsoon']
D1 = weather + ['water_area_km2', 'flood_today']
D2 = [c for c in df.columns if any(c.startswith(k) for k in ('tarbela', 'kabul', 'mangla', 'marala'))] + ['lin_flow_pred']
D4 = ['prev_rain', 'prev_anom', 'cum3', 'sarima_next', 'sarima_gap']
SETS = {'D1': D1, 'D1+D2': D1+D2, 'D1+D4': D1+D4, 'D1+D2+D4 (fused)': D1+D2+D4}
y = df.flood_tomorrow.values; tr = (df.date.dt.year < 2024).values; te = (df.date.dt.year == 2024).values
for c in D2+D4: df[c] = df[c].ffill().bfill()

def lstm_fit_predict(Xtr, ytr, Xte, seed, W=7, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler().fit(Xtr); A = np.vstack([sc.transform(Xtr), sc.transform(Xte)]); n = len(Xtr)
    seq = np.stack([A[i-W+1:i+1] for i in range(W-1, len(A))]); off = W-1
    Str = torch.tensor(seq[:n-off], dtype=torch.float32); ytr_ = torch.tensor(ytr[off:], dtype=torch.float32)
    Ste = torch.tensor(seq[n-off:], dtype=torch.float32)
    class M(nn.Module):
        def __init__(s, d): super().__init__(); s.l = nn.LSTM(d, 32, batch_first=True); s.dr = nn.Dropout(.2); s.o = nn.Linear(32, 1)
        def forward(s, x): h, _ = s.l(x); return s.o(s.dr(h[:, -1])).squeeze(-1)
    m = M(A.shape[1]); opt = torch.optim.Adam(m.parameters(), 1e-3)
    w1 = len(ytr_)/(2*ytr_.sum()+1e-9); w0 = len(ytr_)/(2*(len(ytr_)-ytr_.sum())+1e-9)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Str))
        for i in range(0, len(Str), 32):
            idx = perm[i:i+32]; lo = m(Str[idx]); wt = torch.where(ytr_[idx] > .5, w1, w0)
            loss = (nn.functional.binary_cross_entropy_with_logits(lo, ytr_[idx], reduction='none')*wt).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): pt = torch.sigmoid(m(Ste)).numpy()
    return np.concatenate([np.full(off, np.nan), pt]) if False else pt   # len(Xte) rows aligned to last len(Xte)

def lstm_prob(Xtr, ytr, Xte, seed):
    # pad the test block with the last W-1 training rows so every test day gets a prediction
    W = 7; Xc = np.vstack([Xtr[-(W-1):], Xte]); yc = ytr
    p = lstm_fit_predict(Xtr, ytr, Xc, seed)           # predictions for Xc rows from index W-1 on
    return p[-len(Xte):] if len(p) >= len(Xte) else np.r_[np.full(len(Xte)-len(p), .5), p]
def fit_predict(name, Xtr, ytr, Xte, seed=42):
    if name == 'LR': m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    if name == 'RF': m = RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced', random_state=seed, n_jobs=2).fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    if name == 'XGB': m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8, eval_metric='logloss', random_state=seed, n_jobs=2).fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1]
    if name == 'LSTM': return lstm_prob(Xtr, ytr, Xte, seed)
BASES = ['LR', 'RF', 'XGB', 'LSTM']
def oof_blocks(Xtr, ytr, k=5, seed=42, bases=BASES):
    n = len(Xtr); edges = np.linspace(int(n*.3), n, k+1).astype(int); O = np.full((n, len(bases)), np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        for j, nm in enumerate(bases): O[a:b, j] = fit_predict(nm, Xtr[:a], ytr[:a], Xtr[a:b], seed)   # expanding window: only the past is used
    return O, edges[0]
def stack_predict(F, seed=42, bases=BASES):
    Xtr, Xte, ytr = F[tr], F[te], y[tr]
    O, s0 = oof_blocks(Xtr, ytr, seed=seed, bases=bases)
    meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(np.log(O[s0:]/(1-O[s0:])+1e-9).clip(-12, 12), ytr[s0:])
    T = np.column_stack([fit_predict(nm, Xtr, ytr, Xte, seed) for nm in bases]); Tl = np.log(T/(1-T)+1e-9).clip(-12, 12)
    pte = meta.predict_proba(Tl)[:, 1]; poof = meta.predict_proba(np.log(O[s0:]/(1-O[s0:])+1e-9).clip(-12, 12))[:, 1]
    ths = np.linspace(.05, .95, 91); f1s = [f1_score(ytr[s0:], poof >= t) for t in ths]
    return pte, T, float(ths[int(np.argmax(f1s))]), dict(zip(bases, meta.coef_[0].round(3)))
def metrics(yt, p, th=.5):
    pr = (p >= th).astype(int)
    return dict(acc=accuracy_score(yt, pr), prec=precision_score(yt, pr, zero_division=0), rec=recall_score(yt, pr), f1=f1_score(yt, pr), auc=roc_auc_score(yt, p), prauc=average_precision_score(yt, p), brier=brier_score_loss(yt, p))
def boot(yt, p, th=.5, B=1000, blk=7, seed=0):
    rng = np.random.default_rng(seed); n = len(yt); nb = int(np.ceil(n/blk)); f, a = [], []
    for _ in range(B):
        st = rng.integers(0, n-blk+1, nb); ix = np.concatenate([np.arange(s, s+blk) for s in st])[:n]
        if yt[ix].min() == yt[ix].max(): continue
        f.append(f1_score(yt[ix], p[ix] >= th)); a.append(roc_auc_score(yt[ix], p[ix]))
    return [np.percentile(f, [2.5, 97.5]).round(3).tolist(), np.percentile(a, [2.5, 97.5]).round(3).tolist()]
R = {}; PRED = {}; yte = y[te]
pers = df.flood_today.values[te].astype(float); R['Persistence (tomorrow = today)'] = metrics(yte, pers*.98+.01) | dict(ci=boot(yte, pers))
for sname, cols in SETS.items():
    F = df[cols].values.astype(float)
    for nm in BASES:
        p = fit_predict(nm, F[tr], y[tr], F[te]); R[f'{nm} | {sname}'] = metrics(yte, p) | dict(ci=boot(yte, p)); PRED[f'{nm}|{sname}'] = p
    print(sname, {k.split(' | ')[0]: round(v['f1'], 3) for k, v in R.items() if k.endswith(sname)}, flush=True)
# stacking on every feature set
for sname, cols in SETS.items():
    F = df[cols].values.astype(float); pte, T, th, coef = stack_predict(F)
    R[f'STACK | {sname}'] = metrics(yte, pte) | dict(ci=boot(yte, pte), th_oof=th, coef=coef) | {'tuned': metrics(yte, pte, th)}
    PRED[f'STACK|{sname}'] = pte; print('STACK', sname, round(R[f'STACK | {sname}']['f1'], 3), round(R[f'STACK | {sname}']['auc'], 3), coef, flush=True)
pd.DataFrame(PRED).assign(date=df.date[te].values, y=yte).to_csv('pred_b_test.csv', index=False)
json.dump(R, open('results_b.json', 'w'), indent=1, default=float)
print('done')
