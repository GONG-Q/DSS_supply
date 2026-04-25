import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def load_stable_diffusion(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_attention_slicing: bool = True,
    safety_checker: bool = False
) -> StableDiffusionPipeline:
    """Load Stable Diffusion model"""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype
    ).to(device)
    
    # Configure scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Optimization settings
    if enable_attention_slicing:
        pipe.enable_attention_slicing()
    
    # Disable safety checker
    if not safety_checker:
        pipe.safety_checker = None
    
    return pipe