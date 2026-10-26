import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

	points = np.array(points)
	centroids = np.array(initial_centroids)

	for i in range(max_iterations):
		diff = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
		distances = np.linalg.norm(diff, axis=2)

		assigned = np.argmin(distances, axis=1)
		new_centroids = []

		for label in range(k):
			mask = (assigned == label)
			cluster_mean = np.mean(points[mask], axis=0)
			new_centroids.append(cluster_mean)
		centroids = np.array(new_centroids)
		
	final_centroids = [tuple(c) for c in centroids]	
	return final_centroids