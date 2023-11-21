# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:42:00 2023

@author: basil
"""
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, make_scorer, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import itertools
from sklearn.mixture import GaussianMixture
from preprocessing import standardize
import pandas as pd

scoring = ["adjusted_mutual_info_score", "adjusted_rand_score",
           "completeness_score", "fowlkes_mallows_score",
           "homogeneity_score", "mutual_info_score",
           "normalized_mutual_info_score", "rand_score",
           "v_measure_score"]
scoring = {'adjusted_mutual_info_score': 'adjusted_mutual_info_score',
           'adjusted_rand_score': 'adjusted_rand_score'}

scoring = {
    'Silhouette': make_scorer(silhouette_score),
    'Calinski-Harabasz': make_scorer(calinski_harabasz_score),
    'Davies-Bouldin': make_scorer(davies_bouldin_score),
}

def printText(text, file = None):
    if file == None:
        print(text)
    else:
        file.write(text)

def score(df, clusters, clusterer, min_clust = 0, file = None):
    if len(np.unique(clusters)) > 1:
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, clusters)
        calinski_harabasz = calinski_harabasz_score(df, clusters)
        davies_bouldin = davies_bouldin_score(df, clusters)
        unique_values, counts = np.unique(clusters, return_counts=True)
        if (silhouette_avg > 0.1 and len(unique_values) > min_clust):
            printText(f"\tNb clusters: {len(np.unique(clusters))}\n", file)
            unique_values, counts = np.unique(clusters, return_counts=True)
            if (len(unique_values) < 5):
                for value, count in zip(unique_values, counts):
                    printText(f"\t\t{value}: {count} occurrences\n", file)
            printText(f"\tThe average silhouette_score is: {silhouette_avg}\n"
                  f"\tThe calinski-harabasz score is: {calinski_harabasz} (the higher the better)\n"
                  f"\tThe davies-bouldin score is: {davies_bouldin} (the lower the better)\n",
                  file)        
        return silhouette_avg
    else:
        return None

def plotSilhouette(df, cluster_labels, n_clusters, silhouette_avg):
    plt.xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(df) + (n_clusters + 1) * 10])
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def plotScatterKmeans(df, cluster_labels, n_clusters, clusterer):
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter(
        df[0], df[1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    plt.title("The visualization of the clustered data.")
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")

    plt.show()
    
def plotScatterDBSCAN(df, db):
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = labels == k
    
        xy = df[class_member_mask & core_samples_mask]
        plt.plot(
            xy[0],
            xy[1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )
    
        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[0],
            xy[1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

def clusteringKmeans(df, components = 2):
    pca = PCA(n_components = components) # Keep 2 or 3 components
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)
    df = standardize(df)
    k = np.arange(2, 17)
    k = np.append(k, [25, 35, 45, 100, 250, 500])
    f = open("../results/kmeans_" + str(components) + ".txt", "a")
    
    for n_clusters in k:
        printText(f"For {n_clusters} clusters:\n", file = f)
        
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        cluster_labels = clusterer.fit_predict(df)
        
        silhouette_avg = score(df, cluster_labels, clusterer, file = f)
        
        # Plots
        plotSilhouette(df, cluster_labels, n_clusters, silhouette_avg)
        plotScatterKmeans(df, cluster_labels, n_clusters, clusterer)
    f.close()

def clusteringDBSCAN(df, components = 2):
    pca = PCA(n_components=components) # Best 3, 4
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)
    df = standardize(df)
    
    epslist = [0.1, 0.5, 1, 1.25, 1.5, 2]
    min_sampleslist = np.arange(2, 9)
    all_triplets = list(itertools.product(epslist, min_sampleslist))
    f = open("../results/dbscan_" + str(components) + ".txt", "a")
    for eps, min_samples in all_triplets:
        printText(f"Current parameters: {eps}, {min_samples}", f)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
        labels = clusterer.labels_
        score(df, labels, clusterer, min_clust = 2, file=f)
        plotScatterDBSCAN(df, clusterer)
    f.close()

def clusterHierarchical(df, components = 2):
    pca = PCA(n_components = components)
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)
    df = standardize(df)
    
    f = open("../results/hclust_" + str(components) + ".txt", "a")
    
    n_clusters = [2, 3, 4]
    metrics = ["euclidean", "l1", "l2", "manhattan", "cosine"]
    linkages = ["complete", "average", "single"]
    all_triplets = list(itertools.product(n_clusters, metrics, linkages))
    for k, metric, linkage in all_triplets:
        printText(f"Current parameters: {k}, {metric}, {linkage}", f)
        clusterer = AgglomerativeClustering(n_clusters = k, metric = metric, 
                                            linkage = linkage).fit(df)
        labels = clusterer.labels_
        score(df, labels, clusterer, file = f)
        #plotScatterDBSCAN(df, clusterer)
    
    n_clusters = [2, 3, 4]
    metrics = ["euclidean"]
    linkages = ["ward"]
    all_triplets = list(itertools.product(n_clusters, metrics, linkages))
    for k, metric, linkage in all_triplets:
        printText(f"Current parameters: {k}, {metric}, {linkage}", f)
        clusterer = AgglomerativeClustering(n_clusters = k, metric = metric, 
                                            linkage = linkage).fit(df)
        labels = clusterer.labels_
        score(df, labels, clusterer, file = f)
        #plotScatterDBSCAN(df, clusterer)
    f.close()
    
    
def clusteringGMM(df, components = 2):
    pca = PCA(n_components = components)
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)
    df = standardize(df)
    f = open("../results/gmm_" + str(components) + ".txt", "a")
    
    n_components_range = np.arange(2, 6)
    covariance_types = ['full', 'tied', 'diag', 'spherical']
    all_triplets = list(itertools.product(n_components_range, covariance_types))
    for k, covariance_type in all_triplets:
        printText(f"Current parameters: {k}, {covariance_type}", f)
        clusterer = GaussianMixture(n_components=k, 
                                    covariance_type=covariance_type, 
                                    random_state=42)
        clusterer.fit(df)
        labels = clusterer.predict(df)
        score(df, labels, clusterer, file = f)
        score(df, labels, clusterer)
    f.close()
    
def clusteringSpectral(df, components = 2):
    pca = PCA(n_components = components)
    df = pca.fit_transform(df)
    df = pd.DataFrame(df)
    df = standardize(df)
    
    n_clusters = np.arange(2, 17)
    n_clusters = np.append(n_clusters, [25, 35, 45, 100, 250, 500, 1000, 1500])
    eigen_solvers = ["arpack", "lobpcg", "amg"]
    affinities = ["nearest_neighbors", "rbf"]
    n_neighbors = [2, 5, 10, 25, 50]
    all_triplets = list(itertools.product(n_clusters, eigen_solvers, affinities, n_neighbors))
    for n_cluster, eigen_solver, affinity, n_neighbor in all_triplets:
        f = open("../results/spectral_" + str(components) + ".txt", "a")
        printText(f"Current parameters: {n_cluster}, {eigen_solver}, {affinity}, {n_neighbor}", f)
        printText(f"Current parameters: {n_cluster}, {eigen_solver}, {affinity}, {n_neighbor}")
        clusterer = SpectralClustering(n_clusters=n_cluster, 
                                    eigen_solver=eigen_solver, 
                                    random_state=42,
                                    affinity = affinity,
                                    n_neighbors = n_neighbor)
        try:
            clusterer.fit(df)
            labels = clusterer.labels_
            score(df, labels, clusterer, file = f)
            score(df, labels, clusterer)
        except:
            print("Mem error")
        finally:
            f.close()
    

def cluster(df):
    clusteringSpectral(df, 2)
    clusteringSpectral(df, 3)
    clusteringSpectral(df, 4)
    clusteringSpectral(df, 0.9)
    clusteringSpectral(df, 0.8)
    """
    clusteringGMM(df, 2)
    clusteringGMM(df, 3)
    clusteringGMM(df, 4)
    
    clusteringGMM(df, 0.9)
    clusteringGMM(df, 0.8)
    clusteringDBSCAN(df, 2)
    clusteringDBSCAN(df, 3)
    clusteringDBSCAN(df, 4)
    clusteringDBSCAN(df, 0.9)
    clusteringDBSCAN(df, 0.8)
    #clusteringDBSCAN(df)
    #clusteringGMM(df)
    """