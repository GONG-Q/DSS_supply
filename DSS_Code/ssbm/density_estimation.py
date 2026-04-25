import numpy as np
from scipy.stats import multivariate_normal

def kde_density_estimation(z: np.ndarray, h: float, k: int) -> callable:
    """
    :param z: Projected sensitive embeddings, shape [n, k]
    :param h: Kernel bandwidth
    :param k: Subspace dimension
    :return: Density estimation function \hat{d}(z)
    """
    n = len(z)
    const = 1 / (n * (h ** k) * (2 * np.pi) ** (k / 2))
    
    def density_func(z_query: np.ndarray) -> float:
        """Calculate density for a single query point"""
        distances = np.sum((z_query - z) ** 2, axis=1)
        exp_terms = np.exp(-distances / (2 * h ** 2))
        return const * np.sum(exp_terms)
    
    return density_func

def find_density_peak(z: np.ndarray, density_func: callable) -> np.ndarray:
    """
    Find density peak z_c (sensitive semantic center): z_c = argmax_{z_i} \hat{d}(z_i)
    """
    densities = [density_func(z_i) for z_i in z]
    peak_idx = np.argmax(densities)
    z_c = z[peak_idx]
    return z_c

def negative_log_density_gradient(z_c: np.ndarray, z: np.ndarray, h: float, k: int) -> np.ndarray:
    """
    Calculate negative log-density gradient (Eq.2): z' = z_c - η * ∇log\hat{d}(z_c)/||∇log\hat{d}(z_c)||_2
    :return: Normalized gradient direction vector
    """
    n = len(z)
    # Calculate ∇\hat{d}(z_c)
    delta = z_c - z  # [n, k]
    distances = np.sum(delta ** 2, axis=1)  # [n]
    exp_terms = np.exp(-distances / (2 * h ** 2))  # [n]
    
    grad_d = (1 / (n * (h ** k) * (2 * np.pi) ** (k / 2))) * np.sum(
        exp_terms[:, np.newaxis] * (-delta) / (h ** 2),
        axis=0
    )
    
    # Calculate ∇log\hat{d}(z_c) = ∇\hat{d}(z_c) / \hat{d}(z_c)
    d_zc = kde_density_estimation(z, h, k)(z_c)
    grad_log_d = grad_d / (d_zc + 1e-8)
    
    # Normalize gradient
    grad_log_d_norm = grad_log_d / np.linalg.norm(grad_log_d + 1e-8)
    
    return grad_log_d_norm

def generate_boundary_candidate(z_c: np.ndarray, grad_log_d_norm: np.ndarray, eta: float) -> np.ndarray:
    """
    """
    z_prime = z_c - eta * grad_log_d_norm
    return z_prime