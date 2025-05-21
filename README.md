# MLDR-WiFi
Machine Learning-Driven Region Segmentation for Improved Wi-Fi Fingerprinting in Campus Indoor Localization

Project Overview
This study evaluates four machine learning clustering algorithms—K-Means, DBSCAN, Gaussian Mixture Models (GMM), and Agglomerative Clustering to optimize region formation for Wi-Fi fingerprint-based indoor positioning systems (IPS). By replacing predefined regions with data-driven clusters, the proposed approach significantly improves the accuracy and computational efficiency of the Region-Based K-Nearest Neighbor (RB-KNN) algorithm. Experiments were conducted using a real-world Wi-Fi fingerprint dataset collected in a previous study from Blocks I and J at University West.

Key Features
ML-Driven Region Formation: Clustering algorithms segment Wi-Fi fingerprints into logical regions, reducing search space and computational overhead.
Comparative Analysis: Evaluates K-Means, DBSCAN, GMM, and Agglomerative Clustering against traditional RB-KNN.
Performance Metrics: Measures positioning accuracy (mean error), clustering quality (silhouette score), and computational efficiency (query time).
Visualization Tools: Generates cluster segmentation maps, error distributions, and proximity plots for intuitive result interpretation.

Results
Best Performers:
Agglomerative Clustering achieved the lowest mean positioning error (4.4 m at K=5).
GMM and K-Means reduced positioning errors by 36% and 21%, respectively, compared to RB-KNN.
K-Means had the highest silhouette score, creating the most compact and well separated clusters, offering a clear segmentation for the fingerprint data.

Computational Efficiency:
Sub-millisecond query times (e.g., 0.0003 s for GMM) demonstrated real-time applicability.
Clustering completed in <0.04 s for all algorithms.
DBSCAN Limitations: Poor clustering quality (silhouette score = 0.1429) led to degraded accuracy.

Installation
1.Clone the repository:
https://github.com/wirelessATwest/MLDR-WiFi.git
2.Install dependencies:
pip install requirements.txt
3.MAIN CSV and dataset can be retrieved from the previous study, please visit:
https://github.com/wirelessATwest/PEWFIPS-HV