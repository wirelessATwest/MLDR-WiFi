#!/usr/bin/env python3
# plot_the_results.py
# Generates summary figures and clustering overlays using the full dataset.

import re
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib       import Path
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster       import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture        import GaussianMixture
from sklearn.preprocessing  import StandardScaler
from sklearn.decomposition  import PCA
from sklearn.metrics        import silhouette_score

from common_functions import read_fingerprint_data

# ─────────────────────────── Paths ──────────────────────────────────────────────
BASE     = Path(__file__).parent.resolve()
FP_FILE  = BASE / "fpData-Full.txt"
IMG_DIR  = BASE / "images"
RES_DIR  = BASE / "results"
RES_DIR.mkdir(exist_ok=True)

# ───────────────────────── Plot style ────────────────────────────────────────────
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size':       18,
    'axes.titlesize':  20,
    'axes.labelsize':  18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize':20,
})

# ─────────────────────────── Helper Functions ───────────────────────────────────
ALLOWED_PREFIXES = {
    '70:b3:17:8d:e9','70:b3:17:8e:1c','78:bc:1a:37:7e',
    '48:8b:0a:ca:a8','48:8b:0a:cb:67','48:8b:0a:cb:69'
}

def fingerprint_to_dataframe(fp_dict):
    import pandas as pd
    rows = []
    for coord, rssis in fp_dict.items():
        x, y = map(int, coord.split('_'))
        filt = {b: v for b,v in rssis.items() if b[:14] in ALLOWED_PREFIXES}
        rows.append({'x': x, 'y': y, **filt})
    return pd.DataFrame(rows).fillna(-100)

def group_by_prefix(df):
    bssids = [c for c in df.columns if c not in ('x','y')]
    prefs  = sorted({c[:14] for c in bssids})
    for p in prefs:
        df[p] = df[bssids].filter(like=p).max(axis=1)
    return df[['x','y'] + prefs]



def load_full(fp_path):
    print(f"> Loading fingerprint data from: {fp_path}")
    fp_dict = read_fingerprint_data(str(fp_path))
    df      = fingerprint_to_dataframe(fp_dict)
    df      = group_by_prefix(df)
    X       = df.drop(columns=['x','y']).values
    Xn      = StandardScaler().fit_transform(X)
    return df, Xn

# ─────────────────────────── Plotting Routines ─────────────────────────────────

def plot_bar_chart(summary):
    labels = list(summary.keys())
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*width, [summary[l]['min'] for l in labels], width, label='Min')
    ax.bar(x - 0.5*width, [summary[l]['max'] for l in labels], width, label='Max')
    ax.bar(x + 0.5*width, [summary[l]['std'] for l in labels], width, label='Std Dev')
    ax.bar(x + 1.5*width, [summary[l]['avg'] for l in labels], width, label='Avg')

    ax.set_ylabel('Error (meters)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(RES_DIR/"figure_1_bar_chart.png"), dpi=300)
    plt.close(fig)


def plot_cdf(data_dict, summary):
    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(data_dict)
    for idx, (label, data) in enumerate(data_dict.items()):
        data = np.sort(data)
        cdf  = np.arange(1, len(data)+1) / len(data)
        # plot returns a list of Line2D objects; grab the first one
        line, = ax.plot(data, cdf, label=label)
        # fetch its color
        col = line.get_color()
        # compute the 75th percentile
        q3 = summary[label]['q3']
        # draw the quartile line in the same color, with a tiny jitter so they don't overlap
        jitter = (idx - (n-1)/2) * 0.01
        ax.axvline(q3 + jitter, linestyle='--', color=col, alpha=0.3)

    ax.set_xlabel('Error (meters)')
    ax.set_ylabel('Cumulative Probability')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(str(RES_DIR/"figure_2_cdf.png"), dpi=300)
    plt.close(fig)


def plot_boxplot_grouped(data_dict):
    # Guard against unequal or missing groups
    k5 = [k for k in data_dict if '(k=5)' in k]
    k15 = [k for k in data_dict if '(k=15)' in k]
    data_k5  = [data_dict[k] for k in k5]
    data_k15 = [data_dict[k] for k in k15]
    if len(data_k5) != len(data_k15) or not data_k5:
        print(f"Skipping boxplot: found {len(data_k5)} k=5 and {len(data_k15)} k=15 groups.")
        return
    labels   = [k.split(' ')[0] for k in k5]
    pos_k5  = np.arange(1, len(labels)+1)
    pos_k15 = pos_k5 + (len(labels) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp1 = ax.boxplot(data_k5, positions=pos_k5, patch_artist=True, widths=0.6)
    bp2 = ax.boxplot(data_k15, positions=pos_k15, patch_artist=True, widths=0.6)

    colors  = ['navy','royalblue','limegreen','violet','gold']
    hatches = ['/','x','o','\\',None]

    for bp in (bp1, bp2):
        for patch, color, hatch in zip(bp['boxes'], colors, hatches):
            patch.set_facecolor(color)
            if hatch:
                patch.set_hatch(hatch)

    ax.set_xticks([pos_k5.mean(), pos_k15.mean()])
    ax.set_xticklabels(['K = 5', 'K = 15'])
    ax.set_xlabel('K Value')
    ax.set_ylabel('Error (m)')

    legend_handles = [
        plt.Line2D([0],[0], color='white', markerfacecolor=c, marker='s',
                   label=lbl, markersize=15, markeredgecolor='k')
        for c,lbl in zip(colors, labels)
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.savefig(str(RES_DIR/"figure_3_boxplot.png"), dpi=300)
    plt.close(fig)

# ─────────────────── Clustering & floorplan overlay (full data) ─────────────────

def cluster_and_plot(df, Xn, method, out_png):
    Xf = Xn
    if method == 'kmeans':
        best_k, best_s = 2, -1
        for k in range(2,21):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xf)
            s   = silhouette_score(Xf, lbl)
            if s>best_s:
                best_s, best_k = s, k
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(Xf)

    elif method == 'gmm':
        best_k, best_bic = 2, np.inf
        for k in range(2,21):
            gm  = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3, random_state=42)
            bic = gm.fit(Xf).bic(Xf)
            if bic<best_bic:
                best_bic, best_k = bic, k
        labels = GaussianMixture(n_components=best_k, covariance_type='diag', reg_covar=1e-3, random_state=42).fit_predict(Xf)

    elif method == 'agglomerative':
        best_k, best_s = 2, -1
        for k in range(2,21):
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(Xf)
            s   = silhouette_score(Xf, lbl)
            if s>best_s:
                best_s, best_k = s, k
        labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(Xf)

    elif method == 'dbscan':
        eps_values = [0.8,1.0,1.2,1.6,2.0,2.5,3.0]
        best_sil, best_eps, best_labels = -1.0, None, None
        silhouette_scores = []
        for eps in eps_values:
            lbl_tmp = DBSCAN(eps=eps, min_samples=4).fit_predict(Xf)
            n_clust = len(set(lbl_tmp)) - (1 if -1 in lbl_tmp else 0)
            if n_clust>1:
                s = silhouette_score(Xf, lbl_tmp)
                silhouette_scores.append(s)
                if s>best_sil:
                    best_sil, best_eps, best_labels = s, eps, lbl_tmp
            else:
                silhouette_scores.append(-1)
        # use explicit None check to avoid ambiguous truth value
        if best_labels is not None:
            labels = best_labels
        else:
            labels = DBSCAN(eps=1.0, min_samples=4).fit_predict(Xf)
    else:
        raise ValueError(f"Unknown method: {method}")

    ncl = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(Xf, labels) if ncl>1 else np.nan
    print(f"{method.upper():12s} clusters={ncl:2d} silhouette={sil:.4f}")
    train_df  = df.copy()
    train_df['cluster'] = labels
    centroids = train_df.groupby('cluster')[['x','y']].mean()

    for cid,count in zip(*np.unique(labels, return_counts=True)):
        tag = 'noise' if cid==-1 else f"#{cid}"
        print(f"  → {tag:6s}: {count:3d} pts")
    print("  Centroids:\n", centroids, "\n")

    floor = plt.imread(str(IMG_DIR/"HV I-J Plan 1.png"))
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(floor, origin='upper'); ax.axis('off')

    cmap = plt.get_cmap('tab10' if ncl<=10 else 'tab20', ncl)
    for cid in sorted(set(labels)):
        pts = train_df[train_df.cluster==cid][['x','y']].values
        col = 'lightgrey' if cid==-1 else cmap(int(cid)%cmap.N)
        lbl = 'noise' if cid==-1 else f"Cluster {cid}"
        ax.scatter(pts[:,0], pts[:,1], c=[col], s=30, label=lbl)
        if cid!=-1 and len(pts)>=3:
            try:
                hull = ConvexHull(pts)
                hp   = pts[hull.vertices]
                ax.fill(hp[:,0], hp[:,1], facecolor=col, alpha=0.2, edgecolor='k')
            except QhullError:
                pass

    ax.scatter(centroids.x, centroids.y, c='k', s=100, marker='X', label='Centroid')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1))
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=300)
    plt.close(fig)
    print(f" → saved {out_png}\n")

# ─────────────────────────── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    # --- parse results.txt for ErrorList blocks ---
    stats       = {}
    cur_method  = None
    cur_k       = None

    for line in (RES_DIR/"results.txt").read_text().splitlines():
        m = re.search(r"Algorithm:\s*([^\s,]+)", line)
        if m:
            algo = m.group(1)
            cur_method = algo.split("_")[-1]
            continue

        m = re.search(r"K Value:\s*(\d+)", line)
        if m:
            cur_k = int(m.group(1))
            continue

        m = re.search(r"ErrorList \(meters\):\s*(\[[^\]]*\])", line)
        if m and cur_method and cur_k is not None:
            arr = ast.literal_eval(m.group(1))
            label = f"{cur_method.upper()}-KNN (k={cur_k})"
            stats[label] = [float(x) for x in arr]
            cur_method = None
            cur_k = None

    if stats:
        print("> plotting summary figures into results/")
        summary = {}
        for lbl, errs in stats.items():
            summary[lbl] = {
                'min':  float(np.min(errs)),
                'max':  float(np.max(errs)),
                'std':  float(np.std(errs)),
                'avg':  float(np.mean(errs)),
                'q3':   float(np.percentile(errs, 75))
            }
        plot_bar_chart(summary)
        plot_cdf     (stats, summary)
        plot_boxplot_grouped(stats)
        print(" → figures saved into results/")

    # --- re‐run overlays on full data ---
    df, Xn = load_full(FP_FILE)
    for method in ['kmeans','gmm','agglomerative','dbscan']:
        out_png = RES_DIR / f"cluster_{method}.png"
        cluster_and_plot(df, Xn, method, out_png)
