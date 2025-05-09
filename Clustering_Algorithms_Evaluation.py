# Improved Clustering Evaluation Script for Wi-Fi Fingerprinting
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ALLOWED_PREFIXES = {
    '70:b3:17:8d:e9', '70:b3:17:8e:1c', '78:bc:1a:37:7e',
    '48:8b:0a:ca:a8', '48:8b:0a:cb:67', '48:8b:0a:cb:69'
}

def read_fingerprint_data(file_path):
    fingerprint_data = {}
    fpData = pd.read_csv(file_path)
    for _, row in fpData.iterrows():
        key = f"{row['PosX']}_{row['PosY']}"
        rssi = int(row['RSSI'])
        fingerprint_data.setdefault(key, {})[row['BSSID']] = rssi
    return fingerprint_data

def fingerprint_to_dataframe(fingerprint_data):
    rows = []
    for key, rssi_dict in fingerprint_data.items():
        x, y = map(int, key.split('_'))
        filtered_rssi = {bssid: rssi for bssid, rssi in rssi_dict.items() if bssid[:14] in ALLOWED_PREFIXES}
        row = {'x': x, 'y': y, **filtered_rssi}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.fillna(-100, inplace=True)
    return df

# Load and preprocess data
fp_dict = read_fingerprint_data('fpData-Full.txt')
df = fingerprint_to_dataframe(fp_dict)

bssid_columns = df.columns[2:]
prefixes = [bssid[:14] for bssid in bssid_columns]
unique_prefixes = set(prefixes)
print(f"Unique BSSID prefixes (likely AP count): {len(unique_prefixes)}")

grouped_df = df.copy()
group_map = {}
for bssid in bssid_columns:
    prefix = bssid[:14]
    group_map.setdefault(prefix, []).append(bssid)

for prefix, group in group_map.items():
    grouped_df[prefix] = df[group].max(axis=1)
grouped_df.drop(columns=bssid_columns, inplace=True)

features = grouped_df.drop(columns=['x', 'y'])
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
X_train, X_test = train_test_split(normalized_features, test_size=0.1, random_state=42)

results = []
k_range = range(2, 21)

# --- KMeans ---
best_k, best_score = 2, -1
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_train)
    score = silhouette_score(X_train, labels)
    if score > best_score:
        best_k, best_score = k, score
start = time.time()
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_train)
end = time.time()
results.append({
    'Algorithm': 'KMeans',
    'Best_k': best_k,
    'Clustering Time (s)': round(end - start, 4),
    'Silhouette Score': np.float64(round(silhouette_score(X_train, kmeans.labels_), 4))
})

# --- GMM ---
best_k, lowest_bic = 2, np.inf
for k in k_range:
    gmm = GaussianMixture(n_components=k, covariance_type='diag', reg_covar=1e-3, random_state=42)
    gmm.fit(X_train)
    bic = gmm.bic(X_train)
    if bic < lowest_bic:
        best_k, lowest_bic = k, bic
start = time.time()
gmm = GaussianMixture(n_components=best_k, covariance_type='diag', reg_covar=1e-3, random_state=42).fit(X_train)
end = time.time()
gmm_labels = gmm.predict(X_train)
silhouette = silhouette_score(X_train, gmm_labels) if len(set(gmm_labels)) > 1 else -1.0
results.append({
    'Algorithm': 'GMM',
    'Best_k': best_k,
    'Clustering time (s)': round(end - start, 4),
    'Silhouette Score': np.float64(round(silhouette, 4))
})

# --- Agglomerative ---
best_k, best_score = 2, -1
for k in k_range:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_train)
    score = silhouette_score(X_train, labels)
    if score > best_score:
        best_k, best_score = k, score
start = time.time()
agglomerative = AgglomerativeClustering(n_clusters=best_k).fit(X_train)
end = time.time()
results.append({
    'Algorithm': 'Agglomerative',
    'Best_k': best_k,
    'Clustering Time (s)': round(end - start, 4),
    'Silhouette Score': np.float64(round(silhouette_score(X_train, agglomerative.labels_), 4))
})

# --- DBSCAN with PCA + tuning ---
use_pca = True
X_dbscan = PCA(n_components=5).fit_transform(X_train) if use_pca else X_train

eps_values = [0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0]
best_silhouette = -1.0
best_result = {}

for eps in eps_values:
    start = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=4).fit(X_dbscan)
    end = time.time()
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        silhouette = round(silhouette_score(X_dbscan, labels), 4)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_result = {
                'Algorithm': 'DBSCAN',
                'Best_k': n_clusters,
                'Clustering Time (s)': round(end - start, 4),
                'Silhouette Score': np.float64(silhouette)
            }

if best_result:
    results.append(best_result)
else:
    results.append({
        'Algorithm': 'DBSCAN',
        'Best_k': 1,
        'Clustering Time (s)': 0,
        'Silhouette Score': -1
    })

# Print results in compact format
for r in results:
    print(r)