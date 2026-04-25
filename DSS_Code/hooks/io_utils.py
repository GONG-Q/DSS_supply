import os
import yaml
import numpy as np
import torch
from PIL import Image

def load_config(config_path: str) -> dict:
    """Load yaml configuration"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_embedding(emb: torch.Tensor | np.ndarray, save_path: str):
    """Save embedding vector"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    np.save(save_path, emb)

def save_image(img: Image.Image, save_path: str):
    """Save image"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

def read_prompts_from_txt(file_path: str) -> list:
    """Read prompts from txt file"""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts