#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN_Regions_ML.py

Region-based localization using four clustering algorithms:
KMeans, DBSCAN, Agglomerative, and GMM

For each method, clusters the fingerprint data, assigns test points to the nearest cluster,
and draws the predicted vs. ground-truth lines on the floorplan.
This version now logs results via common_functions.calculate_results() into results/results.txt.
"""
import cv2
import numpy as np
import csv
import pandas as pd
import time
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from common_functions import (
    read_fingerprint_data,
    distance,
    sort_dict_by_rssi,
    calculate_results
)

# Only include access points from the first floor (prefix-based)
ALLOWED_PREFIXES = {
    '70:b3:17:8d:e9',
    '70:b3:17:8e:1c',
    '78:bc:1a:37:7e',
    '48:8b:0a:ca:a8',
    '48:8b:0a:cb:67',
    '48:8b:0a:cb:69'
}

# ------------------------ Core Clustering Integration ------------------------
def fingerprint_to_dataframe(fp_data):
    rows = []
    for key, rssi_dict in fp_data.items():
        x, y = map(int, key.split('_'))
        filtered_rssi = {b: r for b, r in rssi_dict.items() if b[:14] in ALLOWED_PREFIXES}
        row = {'x': x, 'y': y}
        row.update(filtered_rssi)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.fillna(-100, inplace=True)
    return df


def cluster_fingerprint_data(fp_data, method="kmeans"):
    df = fingerprint_to_dataframe(fp_data)

    # group by prefix
    bssid_cols = df.columns[2:]
    grouped = df.copy()
    for b in bssid_cols:
        prefix = b[:14]
        grouped[prefix] = df[bssid_cols].filter(like=prefix).max(axis=1)
    grouped.drop(columns=bssid_cols, inplace=True)

    X = StandardScaler().fit_transform(grouped.drop(columns=['x','y']))
    labels = None

    # choose clustering
    if method == 'kmeans':
        best_k, best_s = 2, -1
        for k in range(2,60):
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            s = silhouette_score(X, lbl)
            if s > best_s:
                best_k, best_s = k, s
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)

    elif method == 'gmm':
        best_k, best_bic = 2, np.inf
        for k in range(2,60):
            g = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3, random_state=42)
            bic = g.fit(X).bic(X)
            if bic < best_bic:
                best_k, best_bic = k, bic
        labels = GaussianMixture(n_components=best_k, covariance_type='diag', reg_covar=1e-3, random_state=42).fit_predict(X)

    elif method == 'agglomerative':
        best_k, best_s = 2, -1
        for k in range(2,60):
            lbl = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            s = silhouette_score(X, lbl)
            if s > best_s:
                best_k, best_s = k, s
        labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(X)

    elif method == 'dbscan':
        from sklearn.decomposition import PCA
        X5 = PCA(n_components=5).fit_transform(X)
        labels = DBSCAN(eps=1.2, min_samples=3).fit_predict(X5)

    else:
        raise ValueError(f"Unknown method: {method}")

    df['cluster'] = labels
    clustered = {}
    for i,row in df.iterrows():
        key = f"{int(row['x'])}_{int(row['y'])}"
        clustered[key] = {'cluster': int(row['cluster']), 'rssi': fp_data[key]}
    return clustered, labels


def GetCandidatePos(online, region_rssi, k):
    cands=[]
    for loc, db in region_rssi.items():
        errs=[]
        t=k
        for b,r in db.items():
            if b in online and t>0:
                errs.append(abs(r-online[b])); t-=1
        if errs:
            cands.append((loc,sum(errs)/len(errs)))
    return min(cands, key=lambda x:x[1]) if cands else (None, float('inf'))


def regionPredictionWithClustering(img, k, fp_data, csv_path, test_dir, method):
    regions,_ = cluster_fingerprint_data(fp_data, method)
    # invert to cluster->loc->rssi
    by_cluster={}  
    for loc,info in regions.items():
        cid=info['cluster']; by_cluster.setdefault(cid,{})[loc]=info['rssi']

    errs=[]; times=[]
    with open(csv_path) as f:
        for gt in csv.reader(f):
            gx,gy=int(gt[1]),int(gt[2])
            testf=os.path.join(test_dir,gt[3]+'.txt')
            online={}
            with open(testf) as tf:
                buf={}; prv=None
                for r in csv.reader(tf,delimiter=';'):
                    if r and r[0]=='WIFI': buf[r[4]]=int(r[5]); prv='WIFI'
                    elif prv=='WIFI':
                        online={b:buf[b] for b in buf if b[:14] in ALLOWED_PREFIXES}; break
            if not online: continue
            online=sort_dict_by_rssi(online)
            # nearest cluster centroid
            best=None; md=float('inf')
            for cid,locs in by_cluster.items():
                for loc in locs:
                    x2,y2=map(int,loc.split('_'))
                    d=distance((gx,gy),(x2,y2))
                    if d<md: md, best = d, cid
            if best is None: continue
            start=time.time()
            pred, _ = GetCandidatePos(online, by_cluster[best], k)
            times.append(time.time()-start)
            if pred:
                px,py=map(int,pred.split('_'))
                cv2.circle(img,(px,py),4,(0,0,0),2)
                cv2.line(img,(gx,gy),(px,py),(0,255,0),2)
                errs.append(distance((gx,gy),(px,py))/35.7)
    sd=np.std(errs)
    print(f"{method.upper()} - Std. dev. error: {sd:.2f} m")
    return errs, times

# ------------------------ Main ------------------------
if __name__=='__main__':
    base_img=cv2.imread('images/HV I-J Plan 1.png')
    fp_data=read_fingerprint_data('fpData-Full.txt')
    csv_path='CSV/Test All.csv'; test_dir='b_All_Tests/'
    k=5; methods=['kmeans','gmm','agglomerative','dbscan']
    os.makedirs('results',exist_ok=True)

    for m in methods:
        img=base_img.copy()
        start=time.time()
        errs, times = regionPredictionWithClustering(img,k,fp_data,csv_path,test_dir,m)
        total_time = time.time()-start
        # write via common_functions
        calculate_results(
            total_time,
            ErrorList=errs,
            PredictionTime=times,
            temp_k=k,
            file_name=f'KNN_Regions_ML_{m}',
            database='fpData-Full.txt'
        )
        out = f'results/positioning_{m}.png'
        cv2.imwrite(out,img)
        print(f"Saved {out}")
