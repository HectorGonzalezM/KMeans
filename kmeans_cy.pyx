# cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as cnp
import cython
import random

@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_distance_cy(cnp.ndarray[cnp.float64_t, ndim=2] data,
                          cnp.ndarray[cnp.float64_t, ndim=2] centroids):
    cdef int n_points = data.shape[0]
    cdef int n_centroids = centroids.shape[0]
    cdef int dim = data.shape[1]
    
    cdef cnp.ndarray[cnp.float64_t, ndim=2] dist_squared = np.empty((n_points, n_centroids), dtype=np.float64)

    cdef int i, j, k
    cdef double diff
    for i in range(n_points):
        for j in range(n_centroids):
            dist_squared[i, j] = 0
            for k in range(dim):
                diff = data[i, k] - centroids[j, k]
                dist_squared[i, j] += diff * diff

    return np.sqrt(dist_squared)

@cython.boundscheck(False)
@cython.wraparound(False)
def assign_points_to_centroids_cy(cnp.ndarray[cnp.float64_t, ndim=2] data,
                                  cnp.ndarray[cnp.float64_t, ndim=2] centroids):
    cdef cnp.ndarray distances = euclidean_distance_cy(data, centroids)
    return np.argmin(distances, axis=1)

@cython.boundscheck(False)
@cython.wraparound(False)
def update_centroids_cy(cnp.ndarray[cnp.float64_t, ndim=2] data, 
                        cnp.ndarray[cnp.int64_t, ndim=1] assignments, 
                        int k):
    cdef int n = data.shape[0]
    cdef int dim = data.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] centroids = np.zeros((k, dim), dtype=np.float64)
    
    cdef cnp.ndarray[cnp.int64_t, ndim=1] counts = np.zeros(k, dtype=np.int64)
    
    cdef int i, j
    for i in range(n):
        counts[assignments[i]] += 1
        for j in range(dim):
            centroids[assignments[i], j] += data[i, j]
    
    for i in range(k):
        if counts[i] == 0:
            centroids[i] = data[random.randint(0, n-1)]
        else:
            for j in range(dim):
                centroids[i, j] /= counts[i]

    return np.asarray(centroids)

def k_means_cy(data, int k, int max_iterations=100, int check_frequency=10):
    data = np.asarray(data, dtype=np.float64)
    cdef int n = data.shape[0]
    cdef int dim = data.shape[1]

    initial_indices = random.sample(range(n), k)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] centroids = data[initial_indices]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] old_centroids = np.empty_like(centroids)

    cdef int iteration
    for iteration in range(max_iterations):
        assignments = assign_points_to_centroids_cy(data, centroids)
        if iteration % check_frequency == 0:
            if np.allclose(old_centroids, centroids, atol=1e-6):
                break
            old_centroids = np.copy(centroids)
        centroids = update_centroids_cy(data, assignments, k)

    clusters = [data[assignments == i].tolist() for i in range(k)]
    return centroids, clusters
