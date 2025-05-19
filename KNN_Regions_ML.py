#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN_Regions_ML.py

Region‐based localization using four clustering algorithms:
KMeans, DBSCAN, Agglomerative, and GMM.

For each method, clusters the fingerprint data, assigns test points to the nearest cluster,
and draws the predicted vs. ground‐truth lines on the floorplan using the same colors
as in plot_the_results.py (tab10/tab20 palette).
Results are logged via common_functions.calculate_results() into results/results.txt.
"""

import os
import csv
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster        import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture         import GaussianMixture
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import silhouette_score

from common_functions import (
    read_fingerprint_data,
    distance,
    sort_dict_by_rssi,
    calculate_results
)

# Only include first‐floor APs (prefix‐based)
ALLOWED_PREFIXES = {
    '70:b3:17:8d:e9','70:b3:17:8e:1c','78:bc:1a:37:7e',
    '48:8b:0a:ca:a8','48:8b:0a:cb:67','48:8b:0a:cb:69'
}

def fingerprint_to_dataframe(fp_data):
    rows = []
    for key, rssi_dict in fp_data.items():
        x, y = map(int, key.split('_'))
        filt = {b: r for b, r in rssi_dict.items() if b[:14] in ALLOWED_PREFIXES}
        row = {'x': x, 'y': y}
        row.update(filt)
        rows.append(row)
    return pd.DataFrame(rows).fillna(-100)

def cluster_fingerprint_data(fp_data, method="kmeans"):
    df = fingerprint_to_dataframe(fp_data)
    # collapse signals by 14‐char prefix
    bssid_cols = [c for c in df.columns if c not in ('x','y')]
    prefixes   = sorted({c[:14] for c in bssid_cols})
    for p in prefixes:
        df[p] = df[bssid_cols].filter(like=p).max(axis=1)
    df.drop(columns=bssid_cols, inplace=True)

    X = StandardScaler().fit_transform(df.drop(columns=['x','y']))

    if method == 'kmeans':
        best_k, best_s = 2, -1
        for k in range(2,21):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            s   = silhouette_score(X, lbl)
            if s > best_s:
                best_k, best_s = k, s
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)

    elif method == 'gmm':
        best_k, best_bic = 2, np.inf
        for k in range(2,21):
            gm  = GaussianMixture(n_components=k, covariance_type='diag',
                                  reg_covar=1e-3, random_state=42).fit(X)
            bic = gm.bic(X)
            if bic < best_bic:
                best_k, best_bic = k, bic
        labels = GaussianMixture(n_components=best_k, covariance_type='diag',
                                 reg_covar=1e-3, random_state=42).fit_predict(X)

    elif method == 'agglomerative':
        best_k, best_s = 2, -1
        for k in range(2,21):
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            s   = silhouette_score(X, lbl)
            if s > best_s:
                best_k, best_s = k, s
        labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(X)

    elif method == 'dbscan':
        eps_values = [0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0]
        best_sil, best_eps, best_labels = -1.0, None, None
        silhouette_scores = []
        for eps in eps_values:
            lbl_tmp = DBSCAN(eps=eps, min_samples=4).fit_predict(X)
            n_clust = len(set(lbl_tmp)) - (1 if -1 in lbl_tmp else 0)
            if n_clust > 1:
                s = silhouette_score(X, lbl_tmp)
                silhouette_scores.append(s)
                if s > best_sil:
                    best_sil, best_eps, best_labels = s, eps, lbl_tmp
            else:
                silhouette_scores.append(-1.0)
        # use explicit None check to avoid ambiguous truth value
        if best_labels is not None:
            labels = best_labels
        else:
            labels = DBSCAN(eps=1.0, min_samples=4).fit_predict(X)
    else:
        raise ValueError(f"Unknown method: {method}")


    clustered = {}
    for (_, row), lbl in zip(df.iterrows(), labels):
        loc = f"{int(row['x'])}_{int(row['y'])}"
        clustered[loc] = {'cluster': int(lbl), 'rssi': fp_data[loc]}
    return clustered, labels

def GetCandidatePos(online, region_rssi, k):
    cands, t = [], k
    for loc, db in region_rssi.items():
        errs, t2 = [], k
        for b,r in db.items():
            if b in online and t2>0:
                errs.append(abs(r-online[b])); t2-=1
        if errs:
            cands.append((loc, sum(errs)/len(errs)))
    return min(cands, key=lambda x: x[1]) if cands else (None, np.inf)

def regionPredictionWithClustering(img, k, fp_data, csv_path, test_dir, method):
    regions, _ = cluster_fingerprint_data(fp_data, method)
    # invert to cluster→ {loc: rssi}
    by_cluster = {}
    for loc,info in regions.items():
        by_cluster.setdefault(info['cluster'], {})[loc] = info['rssi']

    errs, times = [], []
    # determine palette
    ncl  = len(set(info['cluster'] for info in regions.values() if info['cluster']>=0))
    cmap = plt.get_cmap('tab10' if ncl<=10 else 'tab20', ncl)

    # draw RPs
    for loc,info in regions.items():
        x,y = map(int, loc.split('_'))
        cid = info['cluster']
        if cid>=0:
            rgba = cmap(cid % cmap.N)
            bgr  = tuple(int(255*c) for c in reversed(rgba[:3]))
        else:
            bgr  = (200,200,200)
        cv2.circle(img, (x,y), 4, bgr, 2)

    # test points
    with open(csv_path) as f:
        for gt in csv.reader(f):
            gx,gy = int(gt[1]), int(gt[2])
            tf    = os.path.join(test_dir, gt[3]+'.txt')
            online, prv = {}, None
            with open(tf) as fh:
                for r in csv.reader(fh, delimiter=';'):
                    if r and r[0]=='WIFI':
                        online[r[4]] = int(r[5]); prv='WIFI'
                    elif prv=='WIFI':
                        online = {b:online[b] for b in online if b[:14] in ALLOWED_PREFIXES}
                        break
            if not online: continue
            online = sort_dict_by_rssi(online)

            # pick cluster by nearest centroid
            best, md = None, np.inf
            for cid, locs in by_cluster.items():
                for loc in locs:
                    x2,y2 = map(int, loc.split('_'))
                    d     = distance((gx,gy),(x2,y2))
                    if d<md: md, best = d, cid
            if best is None: continue

            t0 = time.time()
            pred,_ = GetCandidatePos(online, by_cluster[best], k)
            times.append(time.time()-t0)

            if pred:
                px,py = map(int, pred.split('_'))
                rgba  = cmap(best % cmap.N)
                bgr_p = tuple(int(255*c) for c in reversed(rgba[:3]))
                cv2.circle(img, (px,py), 4, bgr_p, 2)
                cv2.line(img, (gx,gy), (px,py), (0,255,0), 2)
                errs.append(distance((gx,gy),(px,py))/35.7)

    sd = np.std(errs)
    print(f"{method.upper():<12s} clusters={ncl:2d}  std_err={sd:.2f}m")
    return errs, times

if __name__=='__main__':
    base_img = cv2.imread('images/HV I-J Plan 1.png')
    fp_data  = read_fingerprint_data('fpData-Full.txt')
    csv_path = 'CSV/Test All.csv'
    test_dir = 'b_All_Tests/'
    k        = 15
    methods  = ['kmeans','gmm','agglomerative','dbscan']

    os.makedirs('results', exist_ok=True)

    for m in methods:
        img   = base_img.copy()
        start = time.time()
        errs, times = regionPredictionWithClustering(img, k, fp_data, csv_path, test_dir, m)
        total = time.time() - start

        calculate_results(
            total_time     = total,
            ErrorList      = errs,
            PredictionTime = times,
            temp_k         = k,
            file_name      = f'KNN_Regions_ML_{m}',
            database       = 'fpData-Full.txt'
        )

        out = f'results/k={k}_positioning_{m}.png'
        cv2.imwrite(out, img)

        me = np.mean(errs) if errs else 0.0
        sd = np.std(errs)  if errs else 0.0
        at = np.mean(times) if times else 0.0

        print(f"\n{m.upper()} RESULTS")
        print(f"  Mean error:      {me:.2f} m")
        print(f"  Std. dev error:  {sd:.2f} m")
        print(f"  Avg pred time:   {at*1000:.1f} ms/pt")
        print(f"  Total runtime:   {total:.2f} s")
        print(f"  Saved image:     {out}")
        print("-"*50)
