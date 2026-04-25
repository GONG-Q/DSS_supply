import numpy as np
from sklearn.decomposition import PCA
from src.utils.metric_utils import l2_normalize

def pca_project_embeddings(embeddings: np.ndarray, variance_ratio: float = 0.95) -> tuple:
    """
    :param embeddings: Sensitive sample embeddings, shape [n, d]
    :param variance_ratio: Retained variance ratio
    :return: (Projected embeddings z_i, PCA model, original dimension d, projected dimension k)
    """
    # 1. L2 normalization (Theoretical formula: \tilde{x}_i = x_i/||x_i||_2)
    normed_embeddings = l2_normalize(embeddings)
    
    # 2. PCA projection to M_S subspace (retain 95% variance)
    pca = PCA(n_components=variance_ratio)
    z = pca.fit_transform(normed_embeddings)
    k = z.shape[1]  # Subspace dimension
    
    return z, pca, embeddings.shape[1], k