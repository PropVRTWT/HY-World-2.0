"""
HunyuanImage-3.0 Panorama Pipeline (Qwen-Image-Edit backend)

Usage:
    # Python API — download from HuggingFace
    from pipeline_with_qwen_image import HunyuanPanoPipeline
    pipeline = HunyuanPanoPipeline.from_pretrained(
        lora_path='tencent/HY-World-2.0', lora_subfolder='HY-Pano-2.0')
    output = pipeline('input.png')
    output.save('output_panorama.png')

    # Python API — local path with custom LoRA
    pipeline = HunyuanPanoPipeline.from_pretrained(
        '/path/to/base_model', lora_path='/path/to/lora', lora_subfolder='')
    output = pipeline(
        'input.png',
        prompt='A sunny outdoor scene.',
        seed=42, height=960, width=1952)
    output.save('output_panorama.png')

    # CLI — Basic panorama generation
    python pipeline_with_qwen_image.py --image input.png

    # CLI — Specify prompt, seed and output path
    python pipeline_with_qwen_image.py --image input.png \\
        --prompt "A sunny outdoor scene." --seed 42 --save output_panorama.png

    # CLI — Customize inference steps
    python pipeline_with_qwen_image.py --image input.png \\
        --num-inference-steps 40 --guidance-scale 1.0

    # CLI — Reproducible generation with a fixed seed
    python pipeline_with_qwen_image.py --image input.png --seed 42 --reproduce
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch

from qwen_image import PanoDiffusionPipeline
import enhance

PANO_INSTRUCTION = "Expand this image to a 360-degree equirectangular panorama."

GENERAL_POSITIVE_SUFFIX = " 8k UHD, masterpiece, razor-sharp details."

# The positive prompt is composed from three blocks so that build_positive_prompt()
# can drop or reorder them depending on how much weight the caller's own prompt
# should get. ERP_CORE_INSTRUCTION is the only one that must always survive --
# without it the model stops producing a panorama at all.
ERP_CORE_INSTRUCTION = (
    "Create a seamless **ERP** 360-degree equirectangular panoramic expansion "
    "of the provided image. "
)
IMMERSIVE_FRAMING = (
    "The camera stands inside the space at natural standing eye level, as if the "
    "viewer is physically present in the room. Keep the original subject at its "
    "true scale and distance with natural near-field depth and close surroundings "
    "— do not retreat the viewpoint or shrink the scene into the distance. "
)
STRUCTURAL_FIDELITY = (
    "Preserve the original style, lighting, and fine details seamlessly "
    "throughout the extended areas. Maintain exact window grid alignment, "
    "consistent floor-to-floor height and spacing, straight vertical lines, "
    "and correct perspective convergence with the original structure. Keep all "
    "text and signage sharp, legible, and spelled exactly as shown in the source. "
)

GENERAL_POSITIVE_PREFIX = (
    ERP_CORE_INSTRUCTION + IMMERSIVE_FRAMING + STRUCTURAL_FIDELITY
    + "Extend according to: "
)

# Rewritten from upstream's stock Chinese prompt. Three deliberate changes, all
# targeting observed failures in PropVR interior output:
#
# 1. LANGUAGE. Now English. The stock prompt was 100% Chinese and is enforced at
#    true_cfg_scale=7.5 -- the heaviest text signal in the pipeline -- while the
#    positive prompt is English. Output kept acquiring invented Chinese wall
#    signage. Removing the Chinese-language conditioning is the cheapest test of
#    whether that was the cause. NOTE: this does NOT suppress Chinese present in
#    the INPUT -- that survives via the VAE image conditioning, which is far
#    stronger than text. Only invented signage is targeted, below.
#
# 2. DROPPED the anti-detail terms:
#      杂乱的背景 (cluttered background), 构图混乱 (chaotic composition)
#    At 7.5x these penalise busy, detailed content -- so the model filled the
#    ~240 degrees it has to invent with flat empty walls, which is exactly the
#    "illusion region is all walls" complaint. Blank-wall suppression added
#    instead.
#
# 3. DROPPED 招牌文字错误 (incorrect signage text). Asking for text to be
#    *rendered correctly* implicitly licenses text existing at all.
#
# Previously dropped (kept out): 巨大物体, 巨大建筑, 近景特写, 近景压迫 --
# oversized/close-up penalties that pushed the camera back out of the scene.
GENERAL_NEGATIVE_PROMPT = (
    "low resolution, low quality, blurry, blurry texture, distorted structure, "
    "objects merging together, overly smooth, artificial AI look, deformed faces, "
    "disproportionate scale, cars, vehicles, tree leaves at the top of the frame, "
    "warped architecture, misaligned windows, ghosting, double-exposed facade, "
    "melted building, inconsistent floor spacing, duplicated structural elements, "
    "repeated furniture arrangement, repeating tiled pattern, "
    "blank featureless wall, bare untextured surface, empty monotonous plane, "
    "undetailed filler geometry, unfurnished empty space, "
    "fabricated signage, invented wall text, spurious lettering, "
    "viewpoint retreating, scene too distant, subject too small, empty barren framing."
)

PROMPT_PRIORITY_CHOICES = ("normal", "high", "exclusive")


def build_positive_prompt(prompt: str, priority: str = "normal") -> str:
    """Compose the final positive prompt from the caller's text and the template.

    `priority` controls how much of the text conditioning the caller's own
    prompt gets, relative to the fixed template:

      "normal"    — template first, caller's text spliced in before the suffix
                    (the original behaviour).
      "high"      — caller's text LEADS, and is repeated after the template so
                    it bookends the fixed instructions.
      "exclusive" — caller's text leads and only ERP_CORE_INSTRUCTION +
                    IMMERSIVE_FRAMING survive; the structural-fidelity block is
                    dropped so the caller's wording dominates.

    This is text-side weighting only. The mechanical lever is `guidance_scale`
    (default 1.0 = no classifier-free-guidance amplification of the positive
    prompt, while the negative prompt runs at true_cfg_scale=7.5) -- raise it
    alongside priority="high"/"exclusive" if the prompt still is not landing.
    Repeating the caller's text in "high" mode is a standard diffusion emphasis
    heuristic, not a guarantee.
    """
    if priority not in PROMPT_PRIORITY_CHOICES:
        raise ValueError(
            f"prompt_priority must be one of {PROMPT_PRIORITY_CHOICES}, got {priority!r}"
        )

    user = (prompt or "").strip()

    if priority == "normal":
        return (GENERAL_POSITIVE_PREFIX + user + GENERAL_POSITIVE_SUFFIX).strip()

    if priority == "exclusive":
        parts = ([user] if user else []) + [ERP_CORE_INSTRUCTION, IMMERSIVE_FRAMING]
    else:  # "high"
        parts = ([user] if user else []) + [
            ERP_CORE_INSTRUCTION, IMMERSIVE_FRAMING, STRUCTURAL_FIDELITY,
        ]
        if user:
            parts.append(user)  # bookend the fixed block

    return (" ".join(p.strip() for p in parts if p.strip()) + GENERAL_POSITIVE_SUFFIX).strip()


# ============================================================
# Utility functions
# ============================================================

def circular_blend_edges(image: Image.Image, blend_width: int = 32) -> Image.Image:
    """Blend the left and right edges of an image for seamless panorama."""
    arr = np.array(image)
    for x in range(blend_width):
        arr[:, x, :] = (
            arr[:, -blend_width + x, :] * (1 - x / blend_width)
            + arr[:, x, :] * (x / blend_width)
        )
    return Image.fromarray(arr[:, :-blend_width].astype(np.uint8))


def set_reproducibility(enable: bool, global_seed=None, benchmark=None):
    if enable:
        import random
        random.seed(global_seed)
        import numpy as np
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


# ============================================================
# Pipeline
# ============================================================

class HunyuanPanoPipeline:
    """HunyuanImage panorama pipeline backed by Qwen-Image-Edit.

    Model-level parameters (fixed after loading) are passed to __init__ /
    from_pretrained.  Per-inference parameters are passed to forward / __call__.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
    DEFAULT_LORA_PATH = "tencent/HY-World-2.0"
    DEFAULT_LORA_SUBFOLDER = "HY-Pano-2.0"

    def __init__(self, pipe: PanoDiffusionPipeline):
        """
        Args:
            pipe: Pre-loaded PanoDiffusionPipeline instance.
        """
        self.pipe = pipe

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_ID,
        *,
        lora_path: str = DEFAULT_LORA_PATH,
        lora_subfolder: str = DEFAULT_LORA_SUBFOLDER,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "HunyuanPanoPipeline":
        """Load model weights (and LoRA) and return a ready-to-use pipeline.

        The base model is loaded via diffusers' built-in ``from_pretrained``,
        which handles both local paths and HuggingFace Hub downloads
        automatically.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path
                to the base model directory (i.e. the Qwen-Image-Edit model).
            lora_path: Local path or HuggingFace repo ID for the LoRA weights.
                Pass ``None`` to skip LoRA loading.
            lora_subfolder: Subfolder inside ``lora_path`` (or the HF repo)
                that contains the LoRA weights file.  Ignored when
                ``lora_path`` is ``None``.
            torch_dtype: Torch dtype for the model. Defaults to bfloat16.
        """
        print(f"[Init] Loading base model from {pretrained_model_name_or_path} ...")
        pipe = PanoDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
        ).to("cuda")
        print("[Init] Base model loaded successfully!")

        if lora_path is not None:
            print(f"[Init] Loading LoRA weights from {lora_path} (subfolder={lora_subfolder}) ...")
            pipe.load_lora_weights(
                lora_path,
                subfolder=lora_subfolder,
                weight_name="pytorch_lora_weights.safetensors",
                torch_dtype=torch_dtype,
            )
            print("[Init] LoRA weights loaded successfully!")

        return cls(pipe)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(
        self,
        image,
        *,
        prompt: str = "",
        prompt_priority: str = "normal",
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 960,
        width: int = 1952,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 7.5,
        blend_width: int = 32,
        crop_border: float = 0.0,
        # ---- enhancement (post-generation; see enhance.py) ----
        validate_seam: bool = True,
        seam_threshold: float = 25.0,
        max_seed_retries: int = 1,
        correct_zenith_nadir: bool = False,
        refine_color: bool = False,
    ) -> Image.Image:
        """Run panorama generation and return the blended, optionally-enhanced
        output image.

        Args:
            image: Path to the input image (str or Path).
            prompt: User-provided description composed into the positive template.
            prompt_priority: How much weight the caller's own prompt gets versus
                the fixed template -- "normal" (spliced into the template),
                "high" (leads and is repeated around it), or "exclusive"
                (leads, structural-fidelity block dropped). See
                build_positive_prompt(). Note this is text-side weighting only;
                guidance_scale is the mechanical lever on prompt adherence.
            negative_prompt: Additional negative prompt appended to the default.
            seed: Random seed for reproducibility.
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of diffusion denoising steps.
            guidance_scale: Classifier-free guidance scale.
            true_cfg_scale: True CFG scale for negative-prompt guidance.
            blend_width: Pixel-space edge blending width for final post-process.
            crop_border: Fraction of image border to crop before inference
                (removes compression artefacts on edges).
            validate_seam: Measure the wrap seam after blending; if it's above
                seam_threshold, try a wider re-blend first, then (if that's
                not enough and max_seed_retries > 0) regenerate with the next
                seed and keep whichever attempt has the lowest seam diff.
                Cheap and safe -- no extra model calls unless a seed retry is
                actually needed.
            seam_threshold: Average L1 column-diff above which the seam is
                considered bad (see enhance.measure_seam_diff).
            max_seed_retries: How many extra seeds to try if re-blending alone
                doesn't bring the seam under threshold. 0 disables retrying.
            correct_zenith_nadir: EXPERIMENTAL (default off). Regenerate the
                zenith/nadir caps for geometric straightness. See enhance.py's
                module docstring -- this is a full regeneration of that region,
                not a gentle correction, and can invent new detail.
            refine_color: EXPERIMENTAL (default off). Whole-panorama color/
                exposure refinement pass. Same caveat as correct_zenith_nadir.

        Returns:
            PIL.Image: The generated panorama image, blended and (if enabled)
            enhanced.
        """
        image = str(image)
        if not Path(image).exists():
            raise ValueError(f"Input image does not exist: {image}")

        # Build final prompts
        full_positive = build_positive_prompt(prompt, prompt_priority)
        full_negative = (GENERAL_NEGATIVE_PROMPT + " " + negative_prompt).strip()

        # Crop border to remove compression artefacts
        pil_image = Image.open(image).convert("RGB")
        if crop_border > 0:
            w, h = pil_image.size
            w_crop = int(crop_border * w)
            h_crop = int(crop_border * h)
            pil_image = pil_image.crop((w_crop, h_crop, w - w_crop, h - h_crop))

        def _generate(gen_seed: int) -> Image.Image:
            print("Start generating panorama image...")
            print(f"  Input image:      {image}")
            print(f"  Infer steps:      {num_inference_steps}")
            print(f"  Seed:             {gen_seed}")
            raw = self.pipe(
                image=pil_image,
                prompt=full_positive,
                negative_prompt=full_negative,
                generator=torch.Generator(device="cpu").manual_seed(gen_seed),
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1,
                height=height,
                width=width,
            ).images[0]
            return circular_blend_edges(raw, blend_width)

        result = _generate(seed)

        # ---- Stage 2: seam validation, cheap re-blend, seed retry ----
        if validate_seam:
            result, metrics = enhance.validate_and_fix_seam(
                result, seam_threshold=seam_threshold,
            )
            print(f"  Seam avg diff:    {metrics['seam_avg_before']:.1f}"
                  f" (threshold {seam_threshold})")
            if metrics["reblended"]:
                print(f"  Re-blended ->     {metrics['seam_avg_after']:.1f}")

            best_result, best_avg = result, metrics.get(
                "seam_avg_after", metrics["seam_avg_before"]
            )
            retries_left = max_seed_retries
            next_seed = seed
            while best_avg > seam_threshold and retries_left > 0:
                next_seed += 1
                retries_left -= 1
                print(f"  Seam still bad, retrying with seed={next_seed} "
                      f"({retries_left} retries left)...")
                candidate = _generate(next_seed)
                candidate, cand_metrics = enhance.validate_and_fix_seam(
                    candidate, seam_threshold=seam_threshold,
                )
                cand_avg = cand_metrics.get(
                    "seam_avg_after", cand_metrics["seam_avg_before"]
                )
                print(f"  Retry seam avg diff: {cand_avg:.1f}")
                if cand_avg < best_avg:
                    best_result, best_avg = candidate, cand_avg
            result = best_result

        # ---- Stage 3: EXPERIMENTAL pole correction (off by default) ----
        if correct_zenith_nadir:
            print("  Running EXPERIMENTAL zenith/nadir correction "
                  "(full regeneration of the cap regions -- inspect output)...")
            try:
                result = enhance.correct_poles(
                    self.pipe, result, negative_prompt=full_negative, seed=seed,
                )
            except ImportError as exc:
                print(f"  Skipped pole correction: {exc}")

        # ---- Stage 4: EXPERIMENTAL color/exposure refinement (off by default) ----
        if refine_color:
            print("  Running EXPERIMENTAL color/exposure refinement "
                  "(full regeneration -- inspect output)...")
            result = enhance.refine_colors(
                self.pipe, result, negative_prompt=full_negative, seed=seed,
            )

        return result

    def __call__(self, image, **kwargs) -> Image.Image:
        return self.forward(image, **kwargs)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        "Commandline arguments for running HunyuanPano (Qwen-Image-Edit backend) locally"
    )
    # ---- per-inference ----
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="",
                        help="User prompt composed into the positive template")
    parser.add_argument("--prompt-priority", type=str, default="normal",
                        choices=list(PROMPT_PRIORITY_CHOICES),
                        help="How much weight your --prompt gets vs the fixed "
                             "template: 'normal' splices it in, 'high' leads with "
                             "it and repeats it around the template, 'exclusive' "
                             "leads with it and drops the structural-fidelity "
                             "block. Text weighting only -- raise --guidance-scale "
                             "(default 1.0, i.e. no amplification) if the prompt "
                             "still is not landing hard enough.")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Additional negative prompt appended to the default")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--height", type=int, default=960,
                        help="Height of the generated panorama image.")
    parser.add_argument("--width", type=int, default=1952,
                        help="Width of the generated panorama image (pre-blend; 1952-32=1920).")
    parser.add_argument("--num-inference-steps", type=int, default=40,
                        help="Number of diffusion denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--true-cfg-scale", type=float, default=7.5,
                        help=argparse.SUPPRESS)
    parser.add_argument("--blend-width", type=int, default=32,
                        help="Pixel-space edge blending width for final post-process.")
    parser.add_argument("--crop-border", type=float, default=0.0,
                        help="Fraction of image border to crop before inference.")

    # ---- enhancement (see enhance.py) ----
    parser.add_argument("--no-validate-seam", action="store_false", dest="validate_seam",
                        help="Skip seam measurement/re-blend/retry (on by default).")
    parser.add_argument("--seam-threshold", type=float, default=25.0,
                        help="Avg L1 column-diff above which the seam is considered bad.")
    parser.add_argument("--max-seed-retries", type=int, default=1,
                        help="Extra seeds to try if re-blending doesn't fix a bad seam.")
    parser.add_argument("--correct-zenith-nadir", action="store_true",
                        help="EXPERIMENTAL: regenerate zenith/nadir caps for straight "
                             "lines. Off by default -- see enhance.py's module docstring "
                             "for the hallucination-risk caveat.")
    parser.add_argument("--refine-color", action="store_true",
                        help="EXPERIMENTAL: whole-panorama color/exposure refinement "
                             "pass. Off by default -- same caveat as --correct-zenith-nadir.")

    # ---- model init ----
    parser.add_argument("--pretrained-model-name-or-path", type=str,
                        default=HunyuanPanoPipeline.DEFAULT_MODEL_ID,
                        help="HuggingFace repo ID or local path to the base model")
    parser.add_argument("--lora-path", type=str,
                        default=HunyuanPanoPipeline.DEFAULT_LORA_PATH,
                        help="Local path or HuggingFace repo ID for LoRA weights. "
                             "Pass empty string to skip.")
    parser.add_argument("--lora-subfolder", type=str,
                        default=HunyuanPanoPipeline.DEFAULT_LORA_SUBFOLDER,
                        help="Subfolder inside --lora-path that contains the LoRA weights file.")

    # ---- main-only ----
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the generated image "
                             "(default: <input_stem>_panorama.png)")
    parser.add_argument("--reproduce", action="store_true",
                        help="Whether to reproduce the results (fix all RNGs)")

    return parser.parse_args()


def main(args):
    # Reproducibility
    if args.reproduce:
        set_reproducibility(True, global_seed=args.seed)

    # Build pipeline
    pipeline = HunyuanPanoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        lora_path=args.lora_path if args.lora_path else None,
        lora_subfolder=args.lora_subfolder,
    )

    # Run inference
    output = pipeline(
        args.image,
        prompt=args.prompt,
        prompt_priority=args.prompt_priority,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        blend_width=args.blend_width,
        crop_border=args.crop_border,
        validate_seam=args.validate_seam,
        seam_threshold=args.seam_threshold,
        max_seed_retries=args.max_seed_retries,
        correct_zenith_nadir=args.correct_zenith_nadir,
        refine_color=args.refine_color,
    )

    # Save
    save_path = args.save
    if save_path is None:
        input_stem = Path(args.image).stem
        save_path = f"{input_stem}_panorama.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output.save(save_path)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
