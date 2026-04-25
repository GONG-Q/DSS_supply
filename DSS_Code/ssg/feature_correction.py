import torch

def compute_correction_direction(C_S_l: torch.Tensor, C_N_hat_l: torch.Tensor) -> torch.Tensor:
    """
    Calculate correction direction (Eq.9): d^l = \hat{C}_N^l - C_S^l
    """
    d_l = C_N_hat_l - C_S_l
    return d_l

def compute_optimal_alpha(
    f_l: torch.Tensor,
    C_S_l: torch.Tensor,
    C_N_hat_l: torch.Tensor,
    lambda_balance: float = 0.2
) -> torch.Tensor:
    """
    a* = (f^l - \hat{C}_N^l)^T (C_S^l - \hat{C}_N^l) / [(1+λ) ||C_S^l - \hat{C}_N^l||_2^2]
    """
    numerator = torch.dot(f_l - C_N_hat_l, C_S_l - C_N_hat_l)
    denominator = (1 + lambda_balance) * torch.norm(C_S_l - C_N_hat_l) ** 2
    a_star_l = numerator / (denominator + 1e-8)
    return a_star_l

def correct_feature(
    f_l: torch.Tensor,
    C_S_l: torch.Tensor,
    C_N_hat_l: torch.Tensor,
    lambda_balance: float = 0.2
) -> torch.Tensor:
    """
    """
    # Calculate correction direction
    d_l = compute_correction_direction(C_S_l, C_N_hat_l)
    # Calculate optimal coefficient
    a_star_l = compute_optimal_alpha(f_l, C_S_l, C_N_hat_l, lambda_balance)
    # Correct feature
    f_perp_l = f_l + a_star_l * d_l
    return f_perp_l