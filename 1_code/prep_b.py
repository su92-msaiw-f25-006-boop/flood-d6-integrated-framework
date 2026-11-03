"""Module B: build the fused Dadu daily table (D1 + D2 lags + D4 Sindh rain + SARIMA/Linear forecasts)."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import os; P = os.environ.get('FLOOD_PKG', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')) + '/'
# ---- D1 daily (same rule as the original notebook: mean per day, flood = max)
d1 = pd.read_csv(P+'Dataset_1_Dadu_Sindh/2_kaggle_input/pakistan_flood_complete_dataset.csv'); d1['date'] = pd.to_datetime(d1['date'])
num = [c for c in d1.columns if c not in ['date', 'flood_event']]
daily = d1.groupby('date').agg({**{c: 'mean' for c in num}, 'flood_event': 'max'}).reset_index().sort_values('date').reset_index(drop=True)
# ---- D2: rebuild exact dates from the printed day-month text, walking back from 30-Sep-2025
r = pd.read_csv(P+'Dataset_2_River_Flow/2_kaggle_input/Pakistans_Rivers_Flow_data.csv')
md = pd.to_datetime(r['Date'].str.strip().str.replace('/', '-').str.replace(' ', '') + '-2000', format='%d-%b-%Y')
year = 2025; dates = []; prev = None
for m in md:                                   # file is newest -> oldest
    if prev is not None and m.month == 12 and prev.month == 1: year -= 1
    dates.append(pd.Timestamp(year=year, month=m.month, day=m.day) if not (m.month == 2 and m.day == 29) else pd.Timestamp(year=year, month=2, day=28)); prev = m
r['dt'] = dates; r = r.sort_values('dt').drop_duplicates('dt').set_index('dt')
cols = {'Indus-Tarbela-Inflow': 'tarbela', 'Kabul-Nowshera-Inflow': 'kabul', 'Jehlum-Mangla-Inflow': 'mangla', 'Chenab-Marala-Inflow': 'marala'}
fl = r[list(cols)].rename(columns=cols).asfreq('D').interpolate(limit=5)
print('D2 period', fl.index.min().date(), fl.index.max().date(), 'rows', len(r), 'missing days filled', len(fl)-len(r))
lagc = {}
for c in fl.columns:
    for L in (1, 3, 5, 7, 10): lagc[f'{c}_lag{L}'] = fl[c].shift(L)
    lagc[f'{c}_d3'] = fl[c] - fl[c].shift(3)
lag = pd.DataFrame(lagc)
# Linear-regression flow forecaster: predicts tomorrow's Tarbela inflow from today's flows; fitted on 2021 only
X = pd.concat([fl, fl.shift(1).add_suffix('_p1'), fl.shift(2).add_suffix('_p2')], axis=1); yy = fl['tarbela'].shift(-1)
tr = (X.index.year == 2021) & X.notna().all(axis=1) & yy.notna()
lin = LinearRegression().fit(X[tr], yy[tr]); lag['lin_flow_pred'] = pd.Series(lin.predict(X.ffill().fillna(0)), index=X.index)
lag['lin_flow_pred'] = lag['lin_flow_pred'].where(X.notna().all(axis=1))
# ---- D4 Sindh dekadal rain, anomaly, SARIMA forecast
p4 = pd.read_csv(P+'Dataset_4_Rainfall_KP_Balochistan/2_kaggle_input/ABD_Rainfall_1981_2025_PAK.csv')
p4['dt'] = pd.to_datetime(p4.Date, format='%d-%m-%y', errors='coerce'); p4 = p4[p4.Province == 'Sindh'].dropna(subset=['dt'])
p4['dt'] = p4.dt.where(p4.dt.dt.year <= 2025, p4.dt - pd.DateOffset(years=100))   # %y pivot safety
dk = p4.groupby('dt')['Rainfall - (MM)'].mean().sort_index()
dk = dk[~dk.index.duplicated()]
dkdf = dk.to_frame('rain'); dkdf['dek'] = (dkdf.index.month-1)*3 + np.minimum((dkdf.index.day-1)//10, 2); dkdf['yr'] = dkdf.index.year
norm = dkdf[(dkdf.yr >= 1981) & (dkdf.yr <= 2010)].groupby('dek').rain.mean()
dkdf['norm'] = dkdf.dek.map(norm); dkdf['anom'] = dkdf.rain - dkdf.norm
ls = np.log1p(dkdf.rain.clip(lower=0)); fit_end = dkdf.index < '2022-01-01'
sm = SARIMAX(ls[fit_end].values, order=(1, 0, 1), seasonal_order=(1, 0, 0, 36)).fit(disp=False)
full = sm.apply(ls.values); pred1 = np.expm1(full.get_prediction().predicted_mean)    # one-step-ahead (uses up to previous dekad)
dkdf['sarima_next'] = pred1
dkdf['prev_rain'] = dkdf.rain.shift(1); dkdf['prev_anom'] = dkdf.anom.shift(1); dkdf['cum3'] = dkdf.rain.shift(1).rolling(3).sum()
dkdf['norm_now'] = dkdf.norm
def dek_key(d): return (d.month-1)*3 + min((d.day-1)//10, 2)
dk_feat = dkdf[['prev_rain', 'prev_anom', 'cum3', 'sarima_next', 'norm_now']]
# map each day to its dekad start date
def dekstart(d): return pd.Timestamp(year=d.year, month=d.month, day=(1 if d.day <= 10 else 11 if d.day <= 20 else 21))
daily['dekstart'] = daily.date.map(dekstart)
fused = daily.merge(lag.reset_index().rename(columns={'dt': 'date'}), on='date', how='left').merge(dk_feat.reset_index().rename(columns={'dt': 'dekstart'}), on='dekstart', how='left')
fused['sarima_gap'] = fused['sarima_next'] - fused['norm_now']
fused['flood_today'] = fused.flood_event; fused['flood_tomorrow'] = fused.flood_event.shift(-1)
fused = fused.dropna(subset=['flood_tomorrow']).copy(); fused['flood_tomorrow'] = fused.flood_tomorrow.astype(int)
fused.to_csv('fused_dadu_daily.csv', index=False)
print(fused.shape, 'NaN per column (top):'); print(fused.isna().sum().sort_values(ascending=False).head(8).to_string())
print('sarima one-step R2 on 2022-25 dekads:', np.corrcoef(dkdf.sarima_next[dkdf.index>='2022-01-01'], dkdf.rain[dkdf.index>='2022-01-01'])[0,1]**2)
