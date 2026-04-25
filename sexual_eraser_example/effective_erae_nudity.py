import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionPipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "SD_MODEL_PATH"
DEFAULT_PROMPT_TXT = str(SCRIPT_DIR / "minimal_prompts.txt")

DEFAULT_SENSITIVE_PROMPTS = [
    "nude person, no clothing",
    "naked woman, exposed body",
    "naked man, visible genitalia",
    "full nudity, explicit content",
]

DEFAULT_NORMAL_PROMPTS = [
    "young lady walking by the seaside",
    "adult male sitting on a park bench",
    "female portrait shooting in a photo room",
    "woman strolling along the lake shore",
    "man resting on a lawn in casual wear",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible two-stage intervention script for nude-content suppression."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt-txt", default=DEFAULT_PROMPT_TXT)
    parser.add_argument("--output-root", default=str(SCRIPT_DIR / "demo_outputs"))
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--feature-steps",
        type=int,
        default=15,
        help="Number of denoising steps used during calibration feature extraction.",
    )
    parser.add_argument(
        "--feature-samples",
        type=int,
        default=3,
        help="Number of random seeds used per calibration prompt.",
    )
    parser.add_argument("--lambda-text", type=float, default=0.4)
    parser.add_argument("--lambda-layer", type=float, default=0.4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt used when prompt embeddings are supplied.",
    )
    parser.add_argument(
        "--save-original",
        action="store_true",
        help="Save the pre-intervention image. Disabled by default for blind-review-friendly packaging.",
    )
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_prompts(file_path: str) -> List[str]:
    prompts: List[str] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No prompts found in {file_path}")
    return prompts


def ensure_dirs(output_root: Path) -> Dict[str, Path]:
    original_dir = output_root / "original"
    result_dir = output_root / "result"
    stats_dir = output_root / "stats"
    for path in [output_root, original_dir, result_dir, stats_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_root,
        "original": original_dir,
        "result": result_dir,
        "stats": stats_dir,
    }


def sanitize_path_for_metadata(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if path_value == DEFAULT_MODEL_PATH:
        return DEFAULT_MODEL_PATH
    try:
        return str(path.relative_to(SCRIPT_DIR.parent))
    except ValueError:
        return path.name or path_value


def build_pipeline(model_path: str, device: str) -> StableDiffusionPipeline:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    ).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=False)
    return pipe


def collect_target_layers(pipe: StableDiffusionPipeline) -> List[Tuple[torch.nn.Module, str]]:
    target_layers: List[Tuple[torch.nn.Module, str]] = []

    if hasattr(pipe.unet.mid_block, "attentions"):
        for index, attn in enumerate(pipe.unet.mid_block.attentions):
            target_layers.append((attn, f"mid_block_attn_{index}"))

    for block_index, up_block in enumerate(pipe.unet.up_blocks):
        if hasattr(up_block, "attentions") and len(up_block.attentions) > 0:
            for attn_index, attn in enumerate(up_block.attentions):
                target_layers.append((attn, f"up_block_{block_index}_attn_{attn_index}"))

    if not target_layers:
        raise RuntimeError("No target layers were found in the UNet.")
    return target_layers


def pool_feature(feature: torch.Tensor) -> torch.Tensor:
    if feature.dim() == 3:
        return feature.mean(dim=1)
    if feature.dim() == 4:
        return feature.mean(dim=(2, 3))
    raise ValueError(f"Unsupported feature dimension: {feature.dim()}")


def extract_layer_feature(
    pipe: StableDiffusionPipeline,
    layer: torch.nn.Module,
    prompt: str,
    device: str,
    seeds: Sequence[int],
    feature_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> torch.Tensor:
    feature_bank: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        feat = output[0] if isinstance(output, tuple) else output
        feature_bank.append(pool_feature(feat.detach().float()))

    hook = layer.register_forward_hook(hook_fn)
    try:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                pipe(
                    prompt,
                    num_inference_steps=feature_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
    finally:
        hook.remove()

    if not feature_bank:
        raise RuntimeError(f"Failed to extract features for prompt: {prompt}")
    return torch.cat(feature_bank, dim=0).mean(dim=0)


def encode_prompts(pipe: StableDiffusionPipeline, prompts: Sequence[str], device: str) -> torch.Tensor:
    text_input = pipe.tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        embeddings = pipe.text_encoder(text_input.input_ids)[0]
    return embeddings.float()


def normalize_direction(vector: torch.Tensor) -> torch.Tensor:
    return vector / (vector.norm() + 1e-8)


def compute_text_statistics(
    pipe: StableDiffusionPipeline,
    device: str,
    sensitive_prompts: Sequence[str],
    normal_prompts: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensitive_emb = encode_prompts(pipe, sensitive_prompts, device)
    normal_emb = encode_prompts(pipe, normal_prompts, device)

    sensitive_center = sensitive_emb.mean(dim=(0, 1))
    normal_center = normal_emb.mean(dim=(0, 1))
    direction = normalize_direction(sensitive_center - normal_center)
    return direction, sensitive_center, normal_center


def compute_layer_statistics(
    pipe: StableDiffusionPipeline,
    target_layers: Sequence[Tuple[torch.nn.Module, str]],
    device: str,
    sensitive_prompts: Sequence[str],
    normal_prompts: Sequence[str],
    seeds: Sequence[int],
    feature_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    layer_directions: Dict[str, torch.Tensor] = {}
    layer_centers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for layer, layer_name in target_layers:
        sensitive_feats = [
            extract_layer_feature(
                pipe=pipe,
                layer=layer,
                prompt=prompt,
                device=device,
                seeds=seeds,
                feature_steps=feature_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
            for prompt in sensitive_prompts
        ]
        normal_feats = [
            extract_layer_feature(
                pipe=pipe,
                layer=layer,
                prompt=prompt,
                device=device,
                seeds=seeds,
                feature_steps=feature_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
            for prompt in normal_prompts
        ]

        sensitive_center = torch.stack(sensitive_feats).mean(dim=0)
        normal_center = torch.stack(normal_feats).mean(dim=0)
        direction = normalize_direction(sensitive_center - normal_center)

        layer_centers[layer_name] = (sensitive_center, normal_center)
        layer_directions[layer_name] = direction

    return layer_directions, layer_centers


def save_statistics(
    stats_dir: Path,
    text_direction: torch.Tensor,
    text_sensitive_center: torch.Tensor,
    text_normal_center: torch.Tensor,
    layer_directions: Dict[str, torch.Tensor],
    layer_centers: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    np.save(stats_dir / "text_direction.npy", text_direction.cpu().numpy())
    np.save(stats_dir / "text_sensitive_center.npy", text_sensitive_center.cpu().numpy())
    np.save(stats_dir / "text_normal_center.npy", text_normal_center.cpu().numpy())

    for layer_name, direction in layer_directions.items():
        np.save(stats_dir / f"{layer_name}_direction.npy", direction.cpu().numpy())
        sensitive_center, normal_center = layer_centers[layer_name]
        np.save(stats_dir / f"{layer_name}_sensitive_center.npy", sensitive_center.cpu().numpy())
        np.save(stats_dir / f"{layer_name}_normal_center.npy", normal_center.cpu().numpy())


def apply_closed_form_correction(
    feature: torch.Tensor,
    sensitive_center: torch.Tensor,
    normal_center: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    d = (normal_center - sensitive_center).to(feature.device, dtype=feature.dtype)
    denom = (1.0 + lambda_value) * torch.sum(d * d) + 1e-8

    if feature.dim() == 1:
        v = feature - normal_center.to(feature.device, dtype=feature.dtype)
        a = -torch.sum(v * d) / denom
        return feature + a * d

    if feature.dim() == 2:
        center = normal_center.to(feature.device, dtype=feature.dtype).view(1, -1)
        v = feature - center
        a = -torch.einsum("bc,c->b", v, d) / denom
        return feature + a.unsqueeze(-1) * d.view(1, -1)

    if feature.dim() == 3:
        center = normal_center.to(feature.device, dtype=feature.dtype).view(1, 1, -1)
        v = feature - center
        a = -torch.einsum("bsc,c->bs", v, d) / denom
        return feature + a.unsqueeze(-1) * d.view(1, 1, -1)

    if feature.dim() == 4:
        center = normal_center.to(feature.device, dtype=feature.dtype).view(1, -1, 1, 1)
        v = feature - center
        a = -torch.einsum("bchw,c->bhw", v, d) / denom
        return feature + a.unsqueeze(1) * d.view(1, -1, 1, 1)

    raise ValueError(f"Unsupported feature dimension: {feature.dim()}")


def validate_text_direction(
    text_direction: torch.Tensor,
    sensitive_prompts: Sequence[str],
    normal_prompts: Sequence[str],
    pipe: StableDiffusionPipeline,
    device: str,
) -> Dict[str, float]:
    sensitive_emb = encode_prompts(pipe, sensitive_prompts, device)
    normal_emb = encode_prompts(pipe, normal_prompts, device)
    sensitive_proj = torch.einsum("bld,d->bl", sensitive_emb, text_direction).mean(dim=1)
    normal_proj = torch.einsum("bld,d->bl", normal_emb, text_direction).mean(dim=1)

    return {
        "sensitive_mean_projection": float(sensitive_proj.mean().item()),
        "normal_mean_projection": float(normal_proj.mean().item()),
        "projection_gap": float((sensitive_proj.mean() - normal_proj.mean()).item()),
    }


def make_layer_hook(
    layer_name: str,
    layer_centers: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    lambda_layer: float,
):
    def hook_fn(_module, _inputs, output):
        feat = output[0] if isinstance(output, tuple) else output
        sensitive_center, normal_center = layer_centers[layer_name]
        modified_feat = apply_closed_form_correction(
            feat,
            sensitive_center=sensitive_center,
            normal_center=normal_center,
            lambda_value=lambda_layer,
        )
        if isinstance(output, tuple):
            return (modified_feat,) + output[1:]
        return modified_feat

    return hook_fn


def generate_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    intervene: bool,
    text_centers: Tuple[torch.Tensor, torch.Tensor] = None,
    target_layers: Sequence[Tuple[torch.nn.Module, str]] = (),
    layer_centers: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = None,
    lambda_text: float = 1.0,
    lambda_layer: float = 1.0,
    negative_prompt: str = "",
):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    hooks = []

    prompt_embeds = None
    negative_prompt_embeds = None

    if intervene and text_centers is not None:
        sensitive_center, normal_center = text_centers
        original_prompt_embeds = encode_prompts(pipe, [prompt], pipe.device)
        prompt_embeds = apply_closed_form_correction(
            original_prompt_embeds,
            sensitive_center=sensitive_center,
            normal_center=normal_center,
            lambda_value=lambda_text,
        )
        negative_prompt_embeds = encode_prompts(pipe, [negative_prompt], pipe.device)

    if intervene and layer_centers is not None:
        for layer, layer_name in target_layers:
            if layer_name in layer_centers:
                hooks.append(layer.register_forward_hook(make_layer_hook(layer_name, layer_centers, lambda_layer)))

    try:
        with torch.no_grad():
            if prompt_embeds is not None:
                result = pipe(
                    prompt_embeds=prompt_embeds.to(device=pipe.device, dtype=pipe.text_encoder.dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(device=pipe.device, dtype=pipe.text_encoder.dtype),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
            else:
                result = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=height,
                    width=width,
                )
        return result.images[0]
    finally:
        for hook in hooks:
            hook.remove()


def save_run_metadata(
    output_root: Path,
    args: argparse.Namespace,
    target_layers: Sequence[Tuple[torch.nn.Module, str]],
    validation: Dict[str, float],
) -> None:
    payload = {
        "model_path": sanitize_path_for_metadata(args.model_path),
        "prompt_txt": sanitize_path_for_metadata(args.prompt_txt),
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "feature_steps": args.feature_steps,
        "feature_samples": args.feature_samples,
        "lambda_text": args.lambda_text,
        "lambda_layer": args.lambda_layer,
        "height": args.height,
        "width": args.width,
        "save_original": args.save_original,
        "prompt_count": len(list(target_layers)) if False else None,
        "target_layer_count": len(target_layers),
        "text_direction_validation": validation,
        "closed_form_formula": {
            "direction": "d = C_n - C_s",
            "coefficient": "a = - <f - C_n, d> / ((1 + lambda) * ||d||^2 + eps)",
            "update": "f' = f + a * d",
        },
    }
    payload["prompt_count"] = validation.get("prompt_count", payload["prompt_count"])
    with open(output_root / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_paths = ensure_dirs(Path(args.output_root))
    prompts = load_prompts(args.prompt_txt)
    seeds = [args.seed + index for index in range(args.feature_samples)]

    print(f"Using device: {device}")
    print(f"Loaded {len(prompts)} prompts from {args.prompt_txt}")

    pipe = build_pipeline(args.model_path, device)
    target_layers = collect_target_layers(pipe)
    print(f"Collected {len(target_layers)} target layers.")

    print("Stage 1/2: calibrating text and layer statistics...")
    text_direction, text_sensitive_center, text_normal_center = compute_text_statistics(
        pipe=pipe,
        device=device,
        sensitive_prompts=DEFAULT_SENSITIVE_PROMPTS,
        normal_prompts=DEFAULT_NORMAL_PROMPTS,
    )
    layer_directions, layer_centers = compute_layer_statistics(
        pipe=pipe,
        target_layers=target_layers,
        device=device,
        sensitive_prompts=DEFAULT_SENSITIVE_PROMPTS,
        normal_prompts=DEFAULT_NORMAL_PROMPTS,
        seeds=seeds,
        feature_steps=args.feature_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )
    save_statistics(
        stats_dir=output_paths["stats"],
        text_direction=text_direction,
        text_sensitive_center=text_sensitive_center,
        text_normal_center=text_normal_center,
        layer_directions=layer_directions,
        layer_centers=layer_centers,
    )
    validation = validate_text_direction(
        text_direction=text_direction,
        sensitive_prompts=DEFAULT_SENSITIVE_PROMPTS,
        normal_prompts=DEFAULT_NORMAL_PROMPTS,
        pipe=pipe,
        device=device,
    )
    save_run_metadata(
        output_root=output_paths["root"],
        args=args,
        target_layers=target_layers,
        validation={**validation, "prompt_count": len(prompts)},
    )
    print(f"Text-direction projection gap: {validation['projection_gap']:.4f}")

    print("Stage 2/2: generating intervened images...")
    text_centers = (text_sensitive_center, text_normal_center)
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}] {prompt}")
        image_original = None
        if args.save_original:
            image_original = generate_image(
                pipe=pipe,
                prompt=prompt,
                seed=args.seed,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                intervene=False,
            )
        image_intervened = generate_image(
            pipe=pipe,
            prompt=prompt,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.height,
            width=args.width,
            intervene=True,
            text_centers=text_centers,
            target_layers=target_layers,
            layer_centers=layer_centers,
            lambda_text=args.lambda_text,
            lambda_layer=args.lambda_layer,
            negative_prompt=args.negative_prompt,
        )

        result_name = "single_result.png" if len(prompts) == 1 else f"prompt_{index:03d}_intervened.png"
        intervened_path = output_paths["result"] / result_name
        image_intervened.save(intervened_path)
        if image_original is not None:
            original_path = output_paths["original"] / f"prompt_{index:03d}_original.png"
            image_original.save(original_path)
            print(f"Saved: {original_path.name}, {intervened_path.name}")
        else:
            print(f"Saved: {intervened_path.name}")


if __name__ == "__main__":
    main()
