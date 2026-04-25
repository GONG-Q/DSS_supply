import os
import sys
import time
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dss_run.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import core modules
from src.model.sd_loader import load_stable_diffusion
from src.model.feature_extractor import collect_target_layers, extract_text_embeddings
from src.ssbm.pca_projection import pca_project_embeddings
from src.ssbm.density_estimation import (
    kde_density_estimation, find_density_peak,
    negative_log_density_gradient, generate_boundary_candidate
)
from src.ssbm.safe_anchor_selection import select_top_k_safe_anchors
from src.ssg.reference_pair import build_reference_pairs
from src.ssg.sensitivity_scoring import (
    fuse_normal_center, calculate_sensitivity_score, calculate_balanced_threshold
)
from src.hooks.intervention_hooks import get_layer_intervention_hook
from src.utils.io_utils import load_config, save_image, read_prompts_from_txt, save_embedding

# Configuration paths
BASE_CONFIG_PATH = Path("configs/base_config.yaml")
MODEL_CONFIG_PATH = Path("configs/model_config.yaml")
DEBUG_OUTPUT_PATH = Path("debug/")
RESULTS_OUTPUT_PATH = Path("results/")

class DSSNudeEraser:
    def __init__(self):
        self.base_config = self._load_and_validate_config(BASE_CONFIG_PATH)
        self.model_config = self._load_and_validate_config(MODEL_CONFIG_PATH)
        self.device = self._validate_device(self.base_config["general"]["device"])
        self.seed = self.base_config["general"]["seed"]
        self.pipe = None
        self.reference_pairs = None
        self.threshold_T = None
        self.safe_anchors = None
        self.run_stats = {
            "total_prompts": 0,
            "intervened_prompts": 0,
            "total_time": 0.0,
            "avg_time_per_prompt": 0.0,
            "sensitivity_scores": {},
            "correction_coefficients": {}
        }

    def _load_and_validate_config(self, config_path: Path) -> Dict:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} not found")
        config = load_config(str(config_path))
        return config

    def _validate_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, fallback to CPU")
            return "cpu"
        return device

    def _load_data(self) -> Tuple[List[str], List[str], List[str]]:
        sensitive_prompts = [
            "nude person, no clothing",
            "naked woman, exposed body",
            "naked man, visible genitalia",
            "full nudity, explicit content"
        ]
        
        benign_pool_path = Path("data/benign_pool/coco_benign_prompts.txt")
        if not benign_pool_path.exists():
            raise FileNotFoundError(f"Benign pool file {benign_pool_path} not found")
        benign_pool_prompts = read_prompts_from_txt(str(benign_pool_path))
        
        test_prompts_path = Path("data/test_prompts.txt")
        if not test_prompts_path.exists():
            raise FileNotFoundError(f"Test prompts file {test_prompts_path} not found")
        test_prompts = read_prompts_from_txt(str(test_prompts_path))
        
        return sensitive_prompts, benign_pool_prompts, test_prompts

    def init_model(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed) if self.device == "cuda" else None
        
        model_path = self.model_config["sd"]["model_path"]
        if not Path(model_path).exists():
            raise FileNotFoundError(f"SD model path {model_path} not found")
        
        logger.info(f"Loading Stable Diffusion model on {self.device}")
        self.pipe = load_stable_diffusion(
            model_path=model_path,
            device=self.device,
            dtype=torch.float16 if self.model_config["sd"]["dtype"] == "float16" else torch.float32,
            enable_attention_slicing=self.model_config["sd"]["enable_attention_slicing"],
            safety_checker=self.model_config["sd"]["safety_checker"]
        )

    def run_ssbm(self, sensitive_prompts: List[str], benign_pool_prompts: List[str]) -> List[str]:
        logger.info("Starting SSBM phase")
        sensitive_embeddings = extract_text_embeddings(sensitive_prompts, self.pipe, self.device).cpu().numpy()
        
        z, pca_model, d, k = pca_project_embeddings(
            sensitive_embeddings, self.base_config["ssbm"]["pca_variance_ratio"]
        )
        
        density_func = kde_density_estimation(z, self.base_config["ssbm"]["kde_bandwidth"], k)
        z_c = find_density_peak(z, density_func)
        grad_log_d_norm = negative_log_density_gradient(z_c, z, self.base_config["ssbm"]["kde_bandwidth"], k)
        z_prime = generate_boundary_candidate(z_c, grad_log_d_norm, self.base_config["ssbm"]["gradient_step_size"])
        
        benign_embeddings = extract_text_embeddings(benign_pool_prompts, self.pipe, self.device).cpu().numpy()
        top_k_anchors = select_top_k_safe_anchors(
            z_prime, benign_embeddings, pca_model, self.base_config["ssbm"]["top_k_anchors"]
        )
        
        # Save SSBM debug data
        debug_ssbm = {
            "z_c": z_c.tolist(),
            "z_prime": z_prime.tolist(),
            "top_k_anchor_indices": np.argsort(
                [np.dot(z_prime, pca_model.transform(emb.reshape(1, -1))[0]) for emb in benign_embeddings]
            )[-self.base_config["ssbm"]["top_k_anchors"]:].tolist()
        }
        with open(DEBUG_OUTPUT_PATH / "ssbm_debug.json", "w") as f:
            json.dump(debug_ssbm, f)
        
        self.safe_anchors = benign_pool_prompts[:self.base_config["ssbm"]["top_k_anchors"]]
        return self.safe_anchors

    def run_ssg(self, sensitive_prompts: List[str], benign_pool_prompts: List[str]):
        logger.info("Starting SSG phase")
        target_layers = collect_target_layers(self.pipe)
        
        self.reference_pairs = build_reference_pairs(
            self.pipe, sensitive_prompts, self.safe_anchors, target_layers, self.device,
            self.base_config["general"]["num_samples"]
        )
        
        self.threshold_T = calculate_balanced_threshold(
            self.pipe, benign_pool_prompts[:50], sensitive_prompts, self.reference_pairs,
            "mid_block_attn_0", self.device, self.base_config["ssg"]["epsilon"]
        )
        logger.info(f"Balanced threshold T: {self.threshold_T:.4f}")

    def process_prompt_batch(self, test_prompts: List[str]):
        logger.info(f"Processing {len(test_prompts)} test prompts")
        start_time = time.time()
        target_layers = collect_target_layers(self.pipe)
        mid_layer, mid_layer_name = [x for x in target_layers if x[1] == "mid_block_attn_0"][0]
        
        for idx, prompt in enumerate(test_prompts):
            prompt_start = time.time()
            self.run_stats["total_prompts"] += 1
            
            try:
                f_bar = extract_layer_features(
                    self.pipe, mid_layer, mid_layer_name, prompt, self.device,
                    self.base_config["general"]["num_samples"]
                )
                
                C_S_l, normal_centers = self.reference_pairs[mid_layer_name]
                C_N_hat_l, alpha_i = fuse_normal_center(f_bar, normal_centers)
                S = calculate_sensitivity_score(f_bar, C_S_l, C_N_hat_l, self.base_config["ssg"]["epsilon"])
                self.run_stats["sensitivity_scores"][prompt] = S
                
                # Generate original image
                generator = torch.Generator(device=self.device).manual_seed(self.seed + idx)
                img_original = self.pipe(
                    prompt,
                    num_inference_steps=self.model_config["generation"]["num_inference_steps"],
                    generator=generator,
                    guidance_scale=self.model_config["generation"]["guidance_scale"],
                    height=self.model_config["generation"]["height"],
                    width=self.model_config["generation"]["width"]
                ).images[0]
                save_image(img_original, str(RESULTS_OUTPUT_PATH / "original" / f"prompt_{idx+1}.png"))
                
                # Intervention logic
                if S > self.threshold_T:
                    self.run_stats["intervened_prompts"] += 1
                    hooks = []
                    for layer, layer_name in target_layers:
                        if layer_name in self.reference_pairs:
                            hook = layer.register_forward_hook(
                                get_layer_intervention_hook(
                                    layer_name, self.reference_pairs,
                                    self.base_config["ssg"]["lambda_balance"], C_N_hat_l
                                )
                            )
                            hooks.append(hook)
                    
                    img_intervened = self.pipe(
                        prompt,
                        num_inference_steps=self.model_config["generation"]["num_inference_steps"],
                        generator=generator,
                        guidance_scale=self.model_config["generation"]["guidance_scale"],
                        height=self.model_config["generation"]["height"],
                        width=self.model_config["generation"]["width"]
                    ).images[0]
                    save_image(img_intervened, str(RESULTS_OUTPUT_PATH / "intervened" / f"prompt_{idx+1}.png"))
                    
                    # Save correction coefficients
                    correction_coeffs = {}
                    for layer_name in self.reference_pairs.keys():
                        C_S_l, _ = self.reference_pairs[layer_name]
                        coeff = torch.dot(f_bar - C_N_hat_l, C_S_l - C_N_hat_l) / (
                            (1 + self.base_config["ssg"]["lambda_balance"]) * torch.norm(C_S_l - C_N_hat_l) ** 2
                        )
                        correction_coeffs[layer_name] = coeff.item()
                    self.run_stats["correction_coefficients"][prompt] = correction_coeffs
                    
                    for hook in hooks:
                        hook.remove()
                else:
                    save_image(img_original, str(RESULTS_OUTPUT_PATH / "intervened" / f"prompt_{idx+1}.png"))
                
                prompt_time = time.time() - prompt_start
                logger.info(f"Prompt {idx+1} processed in {prompt_time:.2f}s (S={S:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to process prompt {idx+1}: {str(e)}")
                continue
        
        self.run_stats["total_time"] = time.time() - start_time
        self.run_stats["avg_time_per_prompt"] = self.run_stats["total_time"] / len(test_prompts)

    def save_run_results(self):
        # Save statistics
        with open(RESULTS_OUTPUT_PATH / "run_statistics.json", "w") as f:
            json.dump(self.run_stats, f, indent=2)
        
        # Save config snapshot
        config_snapshot = {
            "base_config": self.base_config,
            "model_config": self.model_config,
            "threshold_T": self.threshold_T
        }
        with open(DEBUG_OUTPUT_PATH / "config_snapshot.json", "w") as f:
            json.dump(config_snapshot, f, indent=2)
        
        logger.info(f"Run statistics saved to {RESULTS_OUTPUT_PATH / 'run_statistics.json'}")
        logger.info(f"Total prompts: {self.run_stats['total_prompts']}")
        logger.info(f"Intervened prompts: {self.run_stats['intervened_prompts']}")
        logger.info(f"Total runtime: {self.run_stats['total_time']:.2f}s")
        logger.info(f"Average time per prompt: {self.run_stats['avg_time_per_prompt']:.2f}s")

    def run(self):
        try:
            # Create output directories
            for dir_path in [DEBUG_OUTPUT_PATH, RESULTS_OUTPUT_PATH / "original", RESULTS_OUTPUT_PATH / "intervened"]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Core pipeline
            sensitive_prompts, benign_pool_prompts, test_prompts = self._load_data()
            self.init_model()
            self.run_ssbm(sensitive_prompts, benign_pool_prompts)
            self.run_ssg(sensitive_prompts, benign_pool_prompts)
            self.process_prompt_batch(test_prompts)
            self.save_run_results()
            
            logger.info("DSS pipeline completed successfully")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            sys.exit(1)

def main():
    eraser = DSSNudeEraser()
    eraser.run()

if __name__ == "__main__":
    main()