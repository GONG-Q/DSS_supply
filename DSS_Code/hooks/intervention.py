import torch
from src.ssg.feature_correction import correct_feature

def get_layer_intervention_hook(
    layer_name: str,
    reference_pairs: dict,
    lambda_balance: float,
    C_N_hat_l: torch.Tensor
):
    """Generate layer intervention hook (replace features with corrected f_perp^l)"""
    C_S_l, _ = reference_pairs[layer_name]
    
    def intervention_hook(module, input, output):
        # Extract original features
        f_l = output[0].clone()
        device = f_l.device
        
        # Adapt to different dimension features
        if f_l.dim() == 3:  # [B, seq_len, C]
            B, seq_len, C = f_l.shape
            # Correct each sequence token
            f_perp_l = torch.zeros_like(f_l)
            for b in range(B):
                for s in range(seq_len):
                    f_perp_l[b, s] = correct_feature(
                        f_l[b, s],
                        C_S_l.to(device).to(f_l.dtype),
                        C_N_hat_l.to(device).to(f_l.dtype),
                        lambda_balance
                    )
        elif f_l.dim() == 4:  # [B, C, H, W]
            B, C, H, W = f_l.shape
            # Correct each spatial position
            f_perp_l = torch.zeros_like(f_l)
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        f_perp_l[b, :, h, w] = correct_feature(
                            f_l[b, :, h, w],
                            C_S_l.to(device).to(f_l.dtype),
                            C_N_hat_l.to(device).to(f_l.dtype),
                            lambda_balance
                        )
        else:
            return output
        
        # Return corrected features
        if isinstance(output, tuple):
            return (f_perp_l,) + output[1:]
        else:
            return f_perp_l
    
    return intervention_hook