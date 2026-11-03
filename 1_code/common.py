"""Shared drawing helpers + REAL data loaders for the v3 block diagrams."""
import json, math, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if 'Inter' in f and 'Display' not in f:
        fm.fontManager.addfont(f)
plt.rcParams['font.family'] = 'Inter'

import os; PKG = os.environ.get('FLOOD_PKG', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')) + '/'
INK = '#1F2937'; GREY = '#6B7280'
PAL = {  # main, dark, tint
    'data':  ('#0EA5A4', '#0B6E6D', '#E6F7F6'),
    'prep':  ('#F59E0B', '#A66106', '#FEF3DC'),
    'fuse':  ('#F43F5E', '#A81F3A', '#FFE9ED'),
    'model': ('#4F6BED', '#2B3FA8', '#E8ECFF'),
    'stack': ('#8B5CF6', '#5B33B5', '#F0E9FF'),
    'out':   ('#10B981', '#0A7552', '#E3F8F0'),
    'gold':  ('#F5B700', '#8A6500', '#FFF6D6'),
}


class Canvas:
    def __init__(self, w, h, dpi=220):
        self.fig, self.ax = plt.subplots(figsize=(w, h), dpi=dpi)
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.R = self.fig.canvas.get_renderer(); self.dpi = dpi

    def rbox(self, x, y, w, h, fc, ec='none', r=.07, lw=.8, alpha=1, z=2, ls='-'):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, ls=ls)
        self.ax.add_patch(p); return p

    def shadow_box(self, x, y, w, h, fc='white', ec='#999', lw=.9, r=.07, z=2):
        self.rbox(x+.025, y-.028, w, h, '#000', alpha=.10, z=z-1, r=r)
        return self.rbox(x, y, w, h, fc, ec, lw=lw, r=r, z=z)

    def text(self, x, y, s, size, color=INK, weight='normal', ha='left', va='center', maxw=None, z=8, ls=1.15, rotation=0, style='normal'):
        t = self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, linespacing=ls, rotation=rotation, fontstyle=style)
        if maxw:
            wi = t.get_window_extent(self.R).width / self.dpi
            if wi > maxw + .005:
                print('TEXT TOO WIDE:', repr(s), round(wi, 2), '>', round(maxw, 2))
        return t

    def poly(self, pts, fc, ec='none', lw=0, z=7, alpha=1):
        self.ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))

    def circle(self, x, y, r, fc, ec='none', lw=0, z=7, alpha=1):
        self.ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))

    def line(self, xs, ys, color, lw=1, z=7, ls='-', alpha=1):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls, alpha=alpha, solid_capstyle='round')

    def arrow(self, x1, y1, x2, y2, color, lw=1.3, ms=8, z=6, style='-|>', ls='-'):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, lw=lw, color=color, zorder=z, ls=ls,
                                          shrinkA=0, shrinkB=0))

    def chevron(self, x, y, color):
        self.ax.add_patch(FancyArrowPatch((x-.09, y), (x+.1, y), arrowstyle='simple,head_width=9,head_length=7,tail_width=3.4',
                                          fc=color, ec='none', zorder=4, mutation_scale=1))

    def rect(self, x, y, w, h, fc, ec='none', lw=0, z=7, alpha=1):
        self.ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))

    def save(self, path):
        self.fig.savefig(path, dpi=self.dpi, facecolor='white'); print('saved', path)


# ---------------------------------------------------------------- real data
ALIAS = {'batagram': 'battagram', 'd. i. khan': 'dera ismail khan', 'diamir': 'diamer', 'islamabad': 'islamabad capital territory',
         'jaffarabad': 'jafarabad', 'kambar shahdad kot': 'qambar shahdadkot', 'killa abdullah': 'qilla abdullah', 'killa saifullah': 'qilla saifullah',
         'mirpur khas': 'mirpurkhas', 'naushahro feroze': 'naushehro feroze', 'sheikhupura': 'sheikhpura', 'umer kot': 'umerkot', 'vehari': 'vihari',
         'chitral lower': 'chitral', 'chitral upper': 'chitral', 'kohistan lower': 'kohistan', 'kohistan upper': 'kohistan', 'kolai palas kohistan': 'kohistan',
         'central karachi': 'karachi', 'east karachi': 'karachi', 'korangi karachi': 'karachi', 'malir karachi': 'karachi', 'south karachi': 'karachi', 'west karachi': 'karachi'}


def load_all():
    D = {}
    t = pd.read_csv(PKG + 'Dataset_5_Terrain_Rain_GEE/2_kaggle_input/pak_district_terrain_rain.csv')
    sd = pd.read_csv(PKG + 'Dataset_5_Terrain_Rain_GEE/2_kaggle_input/pak_district_elev_sd.csv')[['shapeID', 'stdDev']].rename(columns={'stdDev': 'elev_sd'})
    t = t.merge(sd, on='shapeID'); t['key'] = t.shapeName.str.lower().str.strip()
    polys = []
    for i, g in enumerate(t['.geo']):
        gj = json.loads(g)
        rings = [gj['coordinates'][0]] if gj['type'] == 'Polygon' else [p[0] for p in gj['coordinates']]
        for r in rings: polys.append((i, np.array(r)))
    D['terrain'] = t; D['polys'] = polys
    fl = pd.read_csv(PKG + 'Dataset_3_Tehsil_Flood_Extent/2_kaggle_input/District_Level_Data.csv')
    fl['key'] = fl.district_name.str.lower().str.strip().replace(ALIAS)
    g = fl.groupby(['key', 'year']).agg(area=('area_km2', 'sum'), flooded=('flooded_km2', 'sum'), province=('province_name', 'first')).reset_index()
    g['pct'] = g.flooded / g.area * 100
    D['flood'] = g
    # D5 modelling table (same as the notebook) + out-of-fold LR risk
    rows = []
    for y in (2022, 2023, 2024, 2025):
        x = g[g.year == y].merge(t, on='key'); x['rain_total'] = x[f'rain_total_{y}']; x['rain_maxday'] = x[f'rain_maxday_{y}']; rows.append(x)
    d = pd.concat(rows, ignore_index=True); d['serious'] = (d.pct > 10).astype(int)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    F = ['elevation', 'slope', 'hand', 'log_flow_acc', 'cropland_frac', 'built_frac', 'flat_frac', 'elev_sd', 'rain_total', 'rain_maxday']
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')), d[F], d.serious,
                          groups=d.key, cv=GroupKFold(5), method='predict_proba')[:, 1]
    d['oof'] = p
    print('D5 OOF LR AUC (check vs notebook .889):', round(roc_auc_score(d.serious, p), 3), 'rows', len(d), 'floods', int(d.serious.sum()))
    D['d5'] = d
    # D1
    d1 = pd.read_csv(PKG + 'Dataset_1_Dadu_Sindh/2_kaggle_input/pakistan_flood_complete_dataset.csv')
    d1['dt'] = pd.to_datetime(d1['date']); d1 = d1.groupby('dt').agg(pr=('precipitation', 'mean'), fl=('flood_event', 'max')).reset_index().sort_values('dt')
    D['d1'] = d1
    # D2
    r = pd.read_csv(PKG + 'Dataset_2_River_Flow/2_kaggle_input/Pakistans_Rivers_Flow_data.csv', index_col=0)
    D['d2'] = r.iloc[::-1].reset_index(drop=True)
    # D4
    p4 = pd.read_csv(PKG + 'Dataset_4_Rainfall_KP_Balochistan/2_kaggle_input/ABD_Rainfall_1981_2025_PAK.csv')
    D['d4'] = p4
    return D


def minimap(cv, D, cx, cy, size, values_by_idx, cmap, vmin, vmax, dadu=True, lw=.15):
    polys = D['polys']
    allp = np.vstack([p for _, p in polys]); lon0, lon1 = allp[:, 0].min(), allp[:, 0].max(); lat0, lat1 = allp[:, 1].min(), allp[:, 1].max()
    kx = math.cos(math.radians((lat0+lat1)/2)); sc = size / max((lon1-lon0)*kx, lat1-lat0)
    for i, p in polys:
        v = values_by_idx.get(i, np.nan)
        col = '#E5E7EB' if (v is None or (isinstance(v, float) and np.isnan(v))) else cmap((v-vmin)/(vmax-vmin+1e-9))
        pts = [(cx + (x-(lon0+lon1)/2)*kx*sc, cy + (y-(lat0+lat1)/2)*sc) for x, y in p]
        cv.ax.add_patch(Polygon(pts, closed=True, fc=col, ec='white', lw=lw, zorder=7))
    if dadu:
        dx = cx + (67.77-(lon0+lon1)/2)*kx*sc; dy = cy + (26.73-(lat0+lat1)/2)*sc
        cv.circle(dx, dy, .03, '#F43F5E', 'white', .6, z=9)
    return sc


def mini_series(cv, x, y, w, h, ys, color, lw=.9, xs=None, fill=False, z=7):
    ys = np.asarray(ys, float); lo, hi = np.nanmin(ys), np.nanmax(ys)
    yy = y + (ys-lo)/(hi-lo+1e-9)*h
    xx = np.linspace(x, x+w, len(ys)) if xs is None else x + (np.asarray(xs)-np.min(xs))/(np.ptp(xs)+1e-9)*w
    if fill:
        cv.ax.fill_between(xx, y, yy, color=color, alpha=.25, lw=0, zorder=z-1)
    cv.line(xx, yy, color, lw, z=z)
    return xx, yy
