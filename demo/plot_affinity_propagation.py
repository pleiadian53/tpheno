"""
=================================================
Demo of affinity propagation clustering algorithm
=================================================

Reference:
Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007

"""
print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

from sklearn import manifold

# compute pairwise distance
from scipy.spatial import distance
import numpy as np

##############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                            random_state=0)

dimX0 = X.shape


# preference controls the number of clusters
# [note] input preference, meaning how likely a particular input is to become an exemplar
#        A value close to the minimum possible similarity produces fewer classes, 
#        while a value close or larger to the maximum possible similarity, produces many classes
from scipy.spatial import distance
D = distance.cdist(X, X, 'euclidean')
off_diag = ~np.eye(D.shape[0],dtype=bool)  # don't include distances to selves
dx = D[off_diag]
# dx = set(D.flatten())
# dx.remove(0.0)
# dx = np.array(list(dx))
sim_median = np.median(-dx)
sim_max = np.max(-dx)
sim_min = np.min(-dx)
print('verify> (negative euclidean) sim median: %f, min: %f, max: %f' % (sim_median, sim_min, sim_max))
# [log] median similarity: -1.952119
# a.flatten()

##############################################################################
# n_components, n_neighbors = 2, 10
# se = manifold.SpectralEmbedding(n_components=n_components,
#                                 n_neighbors=n_neighbors)
# X = se.fit_transform(X)
# print('verify> dim X %s => %s' % (str(dimX0), str(X.shape)))

# Compute Affinity Propagation
# sim_max: -0.002312 => 264 clusters, sim_min: -5.275261 => 12
af = AffinityPropagation(preference=-50).fit(X)   # default:15, -1:27, -10:8, -25:5, -50:3 clusters, -100:96, -200:155 
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

##############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k

    # [log] cluster_centers_indices[0] -> 160
    cluster_center = X[cluster_centers_indices[k]]
    print('verify> cluster_centers_indices[%s]: %s > center: %s' % (k, str(cluster_centers_indices[k]), str(cluster_center)))

    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
