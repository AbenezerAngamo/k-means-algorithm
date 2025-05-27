import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Mall_Customers.csv")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter       
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def _calculate_wcss(self, X, labels):
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum(np.sum((cluster_points - self.centroids[i])**2, axis=1))
        return wcss

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.labels = np.zeros(X.shape[0])
        self.inertia_ = float('inf')
        num_iterations = 0

        for _ in range(self.max_iter):
            num_iterations += 1
            old_centroids = np.copy(self.centroids)

            self.labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)

            if np.allclose(old_centroids, self.centroids):
                break
        self.inertia_ = self._calculate_wcss(X, self.labels)
        return num_iterations

    def predict(self, X):
        if self.centroids is None:
            raise Exception("Fit the model first.")
        return self._assign_clusters(X)
    

class KMeansPlusPlus(KMeans):
    def _initialize_centroids(self, X):
        
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.array([min([np.sum((c - x)**2) for c in centroids]) for x in X])
            probabilities = distances / distances.sum()
            cumulative_probabilities = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(X[j])
                    break
        return np.array(centroids)
    

class KMeansRandomPartition(KMeans):
    def _initialize_centroids(self, X):
        
        initial_labels = np.random.randint(0, self.n_clusters, X.shape[0])
        initial_centroids = np.array([X[initial_labels == i].mean(axis=0) if np.sum(initial_labels == i) > 0 else X[np.random.randint(X.shape[0])]
                                      for i in range(self.n_clusters)])
        return initial_centroids
    

# K-Means implementation with Elbow Method for optimal K

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.centroids[i]
                                  for i in range(self.n_clusters)])
        return new_centroids

    def _calculate_wcss(self, X, labels):
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum(np.sum((cluster_points - self.centroids[i])**2, axis=1))
        return wcss

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.labels = np.zeros(X.shape[0])
        self.inertia_ = float('inf')
        num_iterations = 0

        for _ in range(self.max_iter):
            num_iterations += 1
            old_centroids = np.copy(self.centroids)

            self.labels = self._assign_clusters(X)
            self.centroids = self._update_centroids(X, self.labels)

            if np.allclose(old_centroids, self.centroids):
                break
        self.inertia_ = self._calculate_wcss(X, self.labels)
        return num_iterations

    def predict(self, X):
        if self.centroids is None:
            raise Exception("Fit the model first.")
        return self._assign_clusters(X)

wcss_values = []
max_k = 10

for k in range(1, max_k + 1):
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    wcss_values.append(kmeans_model.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss_values, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(1, max_k + 1))
plt.grid(True)
plt.show()

print("Interpretation of Elbow Method:")
print("Look for the 'elbow' point in the plot where the rate of decrease in WCSS significantly slows down. This point typically indicates the optimal number of clusters.")