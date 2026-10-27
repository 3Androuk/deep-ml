import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    mu = np.mean(data, axis=0)
    sig = np.std(data, axis=0)

    data = (data - mu) / sig

    cov = np.cov(data, rowvar=False)

    lam, V = np.linalg.eigh(cov)

    idx = np.argmax(np.abs(V), axis=0)
    signs = np.sign(V[idx, np.arange(V.shape[1])])

    V *= signs

    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V = V[:, idx]

    return np.round(V[:, :k], decimals=4)