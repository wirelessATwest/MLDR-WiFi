#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_the_results.py

1) Splits the fingerprint dataset into TRAIN (90%) / TEST (10%) with random_state=42
2) Runs clustering on TRAIN only, exactly as Clustering_Algorithms_Evaluation.py
3) Hyperparameter searches (KMeans k, GMM BIC, Agglo k, DBSCAN eps grid) on TRAIN
4) Prints silhouette, cluster sizes, centroids on TRAIN
5) Overlays TRAIN points on the floorplan, with hulls and centroids, saving PNGs
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from common_functions import read_fingerprint_data

# --- Style settings ---
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})


results_stats = {
    
}

# --- 2. CDF and boxplot helpers ---
def plot_cdf(data_dict, save_path=None):
    plt.figure(figsize=(8,6))
    for label, errs in data_dict.items():
        x = np.sort(errs)
        y = np.arange(1, len(x)+1) / len(x)
        plt.plot(x, y, label=label)
    plt.xlabel('Error (m)')
    plt.ylabel('CDF')
    plt.title('CDF of Positioning Errors')
    plt.legend()
    plt.grid(True)
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_box(data_dict, save_path=None):
    plt.figure(figsize=(8,6))
    labels = list(data_dict)
    data = [data_dict[l] for l in labels]
    sns.boxplot(data=data)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Error (m)')
    plt.title('Boxplot of Positioning Errors')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

# --- 3. Load, assemble, split ---
ALLOWED_PREFIXES = {
    '70:b3:17:8d:e9','70:b3:17:8e:1c','78:bc:1a:37:7e',
    '48:8b:0a:ca:a8','48:8b:0a:cb:67','48:8b:0a:cb:69'
}

def fingerprint_to_dataframe(fp_dict):
    import pandas as pd
    rows = []
    for key, rssi_map in fp_dict.items():
        x,y = map(int, key.split('_'))
        filtered = {b: v for b,v in rssi_map.items() if b[:14] in ALLOWED_PREFIXES}
        row = {'x':x,'y':y}
        row.update(filtered)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(-100)
    return df

def group_by_prefix(df):
    bcols = [c for c in df.columns if c not in ('x','y')]
    for b in bcols:
        p = b[:14]
        df[p] = df[bcols].filter(like=p).max(axis=1)
    # keep only x,y plus unique prefixes sorted
    return df[['x','y'] + sorted({c[:14] for c in bcols})]

def load_and_split(fp_csv):
    fp = read_fingerprint_data(fp_csv)
    df = fingerprint_to_dataframe(fp)
    df = group_by_prefix(df)
    X = df.drop(columns=['x','y']).values
    Xn = StandardScaler().fit_transform(X)
    idx = np.arange(len(df))
    train_idx, _ = train_test_split(idx, test_size=0.1, random_state=42)
    return df, Xn, train_idx

# --- 4. Train‐only clustering & plotting ---
def cluster_and_plot_train(df, Xn, train_idx, method, save_path):
    Xtr = Xn[train_idx]

    if method=='kmeans':
        best_k,best_s=-1,-1
        for k in range(2,21):
            lbl = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Xtr)
            s = silhouette_score(Xtr,lbl)
            if s>best_s: best_s,best_k = s,k
        lbls = KMeans(n_clusters=best_k,random_state=42,n_init=10).fit_predict(Xtr)

    elif method=='gmm':
        best_k,lowbic = 2,np.inf
        for k in range(2,21):
            gm = GaussianMixture(n_components=k,covariance_type='diag',reg_covar=1e-3,random_state=42)
            bic = gm.fit(Xtr).bic(Xtr)
            if bic<lowbic: lowbic,best_k = bic,k
        lbls = GaussianMixture(n_components=best_k,covariance_type='diag',reg_covar=1e-3,random_state=42)\
               .fit_predict(Xtr)

    elif method=='agglomerative':
        best_k,best_s = 2,-1
        for k in range(2,21):
            al = AgglomerativeClustering(n_clusters=k).fit_predict(Xtr)
            s = silhouette_score(Xtr,al)
            if s>best_s: best_s,best_k = s,k
        lbls = AgglomerativeClustering(n_clusters=best_k).fit_predict(Xtr)

    elif method=='dbscan':
        Xdb = PCA(n_components=5).fit_transform(Xtr)
        best_s, best_eps, best_lbls = -1,None,None
        for eps in [0.8,1.0,1.2,1.6,2.0,2.5,3.0]:
            lab = DBSCAN(eps=eps,min_samples=4).fit_predict(Xdb)
            ncl = len(set(lab)) - (1 if -1 in lab else 0)
            if ncl>1:
                s = silhouette_score(Xdb,lab)
                if s>best_s: best_s,best_eps,best_lbls = s,eps,lab
        if best_lbls is None:
            best_eps=1.0
            best_lbls = DBSCAN(eps=best_eps,min_samples=4).fit_predict(Xdb)
        lbls = best_lbls
    else:
        raise ValueError(method)

    # Print
    feat_sil = Xtr if method!='dbscan' else PCA(n_components=5).fit_transform(Xtr)
    if len(set(lbls))>1:
        s = silhouette_score(feat_sil,lbls)
        print(f"{method.upper()}: silhouette = {s:.4f}")
    else:
        print(f"{method.upper()}: silhouette N/A (1 cluster)")
    u,c = np.unique(lbls,return_counts=True)
    print(f"{method.upper()}: cluster sizes:")
    for uu,cc in zip(u,c): print(f"  {uu}: {cc}")
    # centroids
    train_df = df.iloc[train_idx].copy()
    train_df['cluster']=lbls
    cents = train_df.groupby('cluster')[['x','y']].mean()
    print(f"{method.upper()}: centroids:\n{cents}\n")

    # Overlay
    img = plt.imread('images/HV I-J Plan 1.png')
    plt.figure(figsize=(12,10))
    plt.imshow(img,origin='upper'); plt.axis('off')
    cmap = plt.get_cmap('tab10' if len(u)<=10 else 'tab20', len(u))
    for cl in u:
        pts = train_df[train_df.cluster==cl][['x','y']].values
        col = 'lightgrey' if cl==-1 else cmap(cl)
        plt.scatter(pts[:,0],pts[:,1],s=30,c=[col],
                    label=('noise' if cl==-1 else f"C{cl}"))
        if cl!=-1 and len(pts)>=3:
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                plt.fill(hp[:,0],hp[:,1],facecolor=col,alpha=0.2,edgecolor='k')
            except QhullError:
                pass
    plt.scatter(cents.x,cents.y,marker='X',c='k',s=100,label='centroids')
    plt.legend(loc='upper right',bbox_to_anchor=(1.3,1.0))
    plt.tight_layout()
    plt.savefig(save_path,dpi=300)
    plt.close()
    print(f"Saved: {save_path}\n")


if __name__=='__main__':
    # optional error plots
    if results_stats:
        plot_cdf(results_stats,save_path='results/cdf.png')
        plot_box(results_stats,save_path='results/box.png')

    # load+split
    df,Xn,train_idx = load_and_split('fpData-Full.txt')
    for m in ['kmeans','gmm','agglomerative','dbscan']:
        cluster_and_plot_train(df,Xn,train_idx,m,f'results/cluster_{m}.png')
