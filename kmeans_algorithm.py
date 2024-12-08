import numpy as np
import time

# Funzione K-means con inizializzazione casuale
def kmeans_random(X, k, iterations=10, tol=1e-4):
    # Inizializzazione casuale dei centroidi
    start_init = time.time()
    centroids = X.sample(n=k, random_state=42).values
    init_time = time.time() - start_init

    assign_time = 0
    update_time = 0

    for iteration in range(iterations):
        # Tempo di assegnazione dei punti
        start_assign = time.time()
        distances = np.linalg.norm(X.values[:, None] - centroids, axis=2)
        closed = np.argmin(distances, axis=1)
        assign_time += time.time() - start_assign

        # Tempo di ricalcolo dei centroidi
        start_update = time.time()
        new_centroids = np.array([X[closed == j].mean().values if sum(closed == j) > 0 else centroids[j]
                                  for j in range(k)])
        update_time += time.time() - start_update

        # Verifica di convergenza
        if np.allclose(new_centroids, centroids, atol=tol):
            break

        centroids = new_centroids

    return {
        "centroids": centroids,
        "assign_time": assign_time,
        "update_time": update_time,
        "init_time": init_time,
        "total_time": assign_time + update_time + init_time
    }

# Funzione K-means con inizializzazione k-means++
def kmeans_plus_plus(X, k, iterations=10, tol=1e-4):
    # Inizializzazione k-means++
    start_init = time.time()
    centroids = [X.iloc[np.random.choice(len(X))].values]
    for _ in range(1, k):
        distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X.values])
        probs = distances / distances.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        next_centroid = X.iloc[np.searchsorted(cumulative_probs, r)].values
        centroids.append(next_centroid)
    centroids = np.array(centroids)
    init_time = time.time() - start_init

    assign_time = 0
    update_time = 0
    prev_centroids = np.zeros_like(centroids)

    for iteration in range(iterations):
        # Tempo per l'assegnazione dei punti
        start_assign = time.time()
        distances = np.linalg.norm(X.values[:, None] - centroids, axis=2)
        closed = np.argmin(distances, axis=1)
        assign_time += time.time() - start_assign

        # Tempo per il ricalcolo dei centroidi
        start_update = time.time()
        new_centroids = np.array([X[closed == j].mean().values if sum(closed == j) > 0 else centroids[j]
                                  for j in range(k)])
        update_time += time.time() - start_update

        # Verifica convergenza
        if np.allclose(new_centroids, prev_centroids, atol=tol):
            break

        prev_centroids = new_centroids.copy()
        centroids = new_centroids

    return {
        "centroids": centroids,
        "assign_time": assign_time,
        "update_time": update_time,
        "init_time": init_time,
        "total_time": assign_time + update_time + init_time
    }
