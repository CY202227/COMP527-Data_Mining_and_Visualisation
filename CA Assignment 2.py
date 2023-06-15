import numpy as np
import random
import matplotlib.pyplot as plt

with open("./data/dataset", "r") as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(" ")
    data.append([float(x) for x in parts[1:]])
X = np.array(data)


def Euclidean(x, y):
    # Compute the Euclidean distance between x and y
    return np.linalg.norm(x - y)


def kmeans(X, k, max_iters=100):
    random.seed(114514)
    centroids = X[random.sample(range(X.shape[0]), k)]
    for i in range(max_iters):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels


def silhouetteCoefficient(data, centroids):
    n = len(data)
    distances = np.zeros((n, len(centroids)))

    for i, point in enumerate(data):
        for j, center in enumerate(centroids):
            distances[i, j] = Euclidean(point, center)

    cluster_assignments = np.argmin(distances, axis=1)
    silhouette = []

    for i in range(n):
        cluster = cluster_assignments[i]
        other_clusters = set(range(len(centroids))) - {cluster}
        same_cluster_points = [j for j in range(n) if cluster_assignments[j] == cluster and i != j]
        if not same_cluster_points:
            silhouette.append(0)
            continue
        avg_same_cluster = np.mean([Euclidean(data[i], data[j]) for j in same_cluster_points])
        avg_other_clusters = [np.mean([Euclidean(data[i], data[j]) for j in range(n) if cluster_assignments[j] == other_cluster]) for other_cluster in other_clusters]
        min_avg_other_clusters = np.min(avg_other_clusters)
        silhouette.append((min_avg_other_clusters - avg_same_cluster) / max(avg_same_cluster, min_avg_other_clusters))

    return np.mean(silhouette)


def silhouette_score(X, labels):
    n = len(X)

    unique_labels = np.unique(labels)
    centroids = [X[labels == i].mean(axis=0) for i in unique_labels]

    return np.mean(silhouetteCoefficient(X, centroids))


def kmeans_pp(X, k, max_iters=100):
    # Initialize centroids
    random.seed(114514)
    centroids = [X[random.randint(0, X.shape[0] - 1)]]
    for temp in range(k - 1):
        distances = np.array([min([np.linalg.norm(x - c) ** 2 for c in centroids]) for x in X])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = random.random()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                i = j
                break
        centroids.append(X[i])
    centroids = np.array(centroids)

    # Run K-means with the initialized centroids
    for i in range(max_iters):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels


silhouette_scores_kmeans = []
silhouette_scores_kmeans_pp = []
for k in range(2, 10):
    labels1 = kmeans(X, k)
    score1 = silhouette_score(X, labels1)

    print(f"Silhouette coefficient for k={k} with K-means: {score1:.4f}")

    silhouette_scores_kmeans.append(score1)


plt.plot(range(2, 10), silhouette_scores_kmeans, "-bo")
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for K-means')
plt.show()

for k in range(2, 10):
    labels2 = kmeans_pp(X, k)
    score2 = silhouette_score(X, labels2)

    print(f"Silhouette coefficient for k={k} with K-means++: {score2:.4f}")

    silhouette_scores_kmeans_pp.append(score2)


plt.plot(range(2, 10), silhouette_scores_kmeans_pp, "-go")
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for K-means++')

plt.show()


def bisecting_kmeans(X, k, max_iters=100):
    if k <= 1:
        return np.zeros(X.shape[0], dtype=int)

    clusters = [X]
    while len(clusters) < k:
        largest_cluster_idx = np.argmax([len(cluster) for cluster in clusters])
        largest_cluster = clusters[largest_cluster_idx]

        labels = kmeans(largest_cluster, 2, max_iters)
        new_clusters = [largest_cluster[labels == i] for i in range(2)]

        del clusters[largest_cluster_idx]
        clusters.extend(new_clusters)

    final_labels = np.zeros(X.shape[0], dtype=int)
    for i, cluster in enumerate(clusters):
        indices = np.where(np.all(X[:, None] == cluster, axis=2))[1]
        final_labels[indices] = i

    return final_labels


silhouette_scores_bisecting_kmeans = []
for k in range(2, 10):
    labels3 = bisecting_kmeans(X, k)
    score3 = silhouette_score(X, labels3)

    print(f"Silhouette coefficient for k={k} with Bisecting K-means: {score3:.4f}")

    silhouette_scores_bisecting_kmeans.append(score3)

plt.plot(range(2, 10), silhouette_scores_bisecting_kmeans, "-ro")
plt.xlabel("k")
plt.ylabel("Silhouette Coefficient")
plt.title("Silhouette Coefficient for Bisecting K-means")
#plt.title("Comparing the different clusterings")
plt.show()
