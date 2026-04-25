import numpy as np
import torch

def l2_normalize(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """L2 normalization"""
    if isinstance(x, np.ndarray):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    elif isinstance(x, torch.Tensor):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """Numpy cosine similarity"""
    x1_norm = l2_normalize(x1)
    x2_norm = l2_normalize(x2)
    return np.dot(x1_norm, x2_norm)

def cosine_similarity_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """PyTorch cosine similarity"""
    x1_norm = l2_normalize(x1)
    x2_norm = l2_normalize(x2)
    return torch.dot(x1_norm, x2_norm)