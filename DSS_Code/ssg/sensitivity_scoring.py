import torch
import numpy as np
from src.utils.metric_utils import cosine_similarity_torch

def fuse_normal_center(
    f_bar: torch.Tensor,
    normal_centers: list[torch.Tensor]
) -> torch.Tensor:
    """
    e_i = <\bar{f}, C_Ni^l>, α_i = exp(e_i)/sum(exp(e_j)), \hat{C}_N^l = sum(α_i C_Ni^l)
    """
    # Calculate e_i = inner product
    e_i = [torch.dot(f_bar, C_Ni) for C_Ni in normal_centers]
    e_i = torch.stack(e_i)
    
    # Calculate attention weights α_i
    alpha_i = torch.softmax(e_i, dim=0)
    
    # Fuse normal centers
    C_N_hat_l = sum(alpha_i[i] * normal_centers[i] for i in range(len(normal_centers)))
    
    return C_N_hat_l, alpha_i

def calculate_sensitivity_score(
    f_bar: torch.Tensor,
    C_S_l: torch.Tensor,
    C_N_hat_l: torch.Tensor,
    epsilon: float = 1e-6
) -> float:
    """
    """
    sim_S = cosine_similarity_torch(f_bar, C_S_l)
    sim_N = cosine_similarity_torch(f_bar, C_N_hat_l)
    S = sim_S / (sim_S + sim_N + epsilon)
    return S.item()

def calculate_balanced_threshold(
    pipe,
    normal_prompts: list,
    sensitive_prompts: list,
    reference_pairs: dict,
    target_layer_name: str,
    device: str,
    epsilon: float = 1e-6
) -> float:
    """
      """
    C_S_l, normal_centers = reference_pairs[target_layer_name]
    
    # Calculate maximum sensitivity score S_N_max for normal prompts
    normal_scores = []
    for prompt in normal_prompts:
        f_bar = extract_layer_features(pipe, *reference_pairs[target_layer_name][0], prompt, device, 1)
        C_N_hat_l, _ = fuse_normal_center(f_bar, normal_centers)
        S = calculate_sensitivity_score(f_bar, C_S_l, C_N_hat_l, epsilon)
        normal_scores.append(S)
    S_N_max = max(normal_scores)
    
    # Calculate minimum sensitivity score S_S_min for sensitive prompts
    sensitive_scores = []
    for prompt in sensitive_prompts:
        f_bar = extract_layer_features(pipe, *reference_pairs[target_layer_name][0], prompt, device, 1)
        C_N_hat_l, _ = fuse_normal_center(f_bar, normal_centers)
        S = calculate_sensitivity_score(f_bar, C_S_l, C_N_hat_l, epsilon)
        sensitive_scores.append(S)
    S_S_min = min(sensitive_scores)
    
    # Calculate threshold T
    T = (S_N_max + S_S_min) / 2
    return T