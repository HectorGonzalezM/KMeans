import random

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def assign_points_to_centroids(data, centroids):
    """Assign each point to the nearest centroid."""
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        nearest_centroid_index = distances.index(min(distances))
        clusters[nearest_centroid_index].append(point)
    return clusters

def update_centroids(clusters):
    """Update the centroid of each cluster."""
    new_centroids = []
    for cluster in clusters:
        if cluster:  # avoid division by zero
            new_centroids.append([sum(values) / len(cluster) for values in zip(*cluster)])
        else:  # if a cluster ended up empty, skip it
            new_centroids.append([])
    return new_centroids

def k_means(data, k, max_iterations=100):
    # Initialize centroids randomly from data points
    centroids = random.sample(data, k)
    old_centroids = [[] for _ in range(k)]  # Dummy old centroids for first comparison

    iterations = 0
    while not all(old == new for old, new in zip(old_centroids, centroids)) and iterations < max_iterations:
        clusters = assign_points_to_centroids(data, centroids)
        old_centroids = centroids
        centroids = update_centroids(clusters)
        iterations += 1

    return centroids, clusters

if __name__ == "__main__":
    data_size = 10000
    dims = 2
    k = 10 
    data = [[random.random() for _ in range(dims)] for _ in range(data_size)]
    max_iterations = 100
    
    # Running k-means
    k_means(data, k, max_iterations)