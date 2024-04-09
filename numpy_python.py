import numpy as np
import numexpr as ne
import random

def euclidean_distance_numpy(data, centroids):
    # Use broadcasting and memory-efficient operations with Numexpr
    diff = np.expand_dims(data, 1) - np.expand_dims(centroids, 0)
    dist_squared = ne.evaluate("sum(diff ** 2, axis=2)")
    return ne.evaluate("sqrt(dist_squared)")

def assign_points_to_centroids_numpy(data, centroids):
    distances = euclidean_distance_numpy(data, centroids)
    return np.argmin(distances, axis=1)

def update_centroids_numpy(data, assignments, k):
    return np.array([data[assignments == i].mean(axis=0) if np.any(assignments == i) else random.choice(data) for i in range(k)])

def k_means_np(data, k, max_iterations=100, check_frequency=10):
    data = np.array(data)
    n = data.shape[0]
    # Initialize centroids randomly from data points
    initial_indices = random.sample(range(n), k)
    centroids = data[initial_indices]
    old_centroids = np.empty_like(centroids)

    for iteration in range(max_iterations):
        assignments = assign_points_to_centroids_numpy(data, centroids)
        if iteration % check_frequency == 0:  # Now 'check_frequency' is defined
            # Only check for convergence every 'check_frequency' iterations to reduce computation
            if np.allclose(old_centroids, centroids, atol=1e-6):
                break
            old_centroids = np.copy(centroids)
        centroids = update_centroids_numpy(data, assignments, k)

    clusters = [data[assignments == i].tolist() for i in range(k)]
    return centroids, clusters

if __name__ == "__main__":
    data_size = 10000
    dims = 2
    k = 10
    data = np.random.rand(data_size, dims)
    max_iterations = 100

    # Running k-means with numpy
    k_means_np(data, k, max_iterations)