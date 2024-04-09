from kmeans_cy import k_means_cy
import numpy as np

if __name__ == "__main__":
    data_size = 10000
    dims = 2
    k = 10
    data = np.random.rand(data_size, dims)
    max_iterations = 100

    # Running k-means with numpy
    k_means_cy(data, k, max_iterations)