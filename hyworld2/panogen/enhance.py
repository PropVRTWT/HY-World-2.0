"""Post-generation enhancement for HY-Pano panoramas.

Three pieces, in increasing order of risk:

  1. measure_seam_diff / reblend_final_edges / validate_and_fix_seam
     Cheap, deterministic, no model calls. Safe to run on every generation.

  2. correct_poles (EXPERIMENTAL)
     Reprojects the zenith/nadir caps to rectilinear views and regenerates them
     with a straight-line-focused prompt, to fix the projection-warp artifact
     seen in ceiling/floor grids near the poles (waviness that's invisible in
     the flat ERP thumbnail but obvious when a viewer looks straight up/down).

  3. refine_colors (EXPERIMENTAL)
     Whole-panorama pass to even out exposure/color across the seam.

IMPORTANT CAVEAT for (2) and (3): PanoDiffusionPipeline (qwen_image/
pipeline_qwen_pano.py) has no `strength`/denoising-control parameter -- every
call is a FULL generation from noise conditioned on the input image, not a
gentle img2img refinement. There is no "light touch" mode. That means both
carry real risk of reintroducing the same hallucination failure modes
(melting, repetition, invented content) the negative prompt fights elsewhere
in this file, rather than just correcting what's already there. Both are OFF
by default in HunyuanPanoPipeline.forward() (see pipeline_with_qwen_image.py)
and should be verified against real output before relying on them.

correct_poles() also depends on `py360convert` (not a base requirement of this
repo) and is untested against a live GPU/model in this change -- the
equirectangular<->perspective calls follow py360convert's published API
(e2p/p2e with fov_deg/u_deg/v_deg/out_hw), but verify on a real run before
trusting it in production.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# ============================================================
# 1. Seam measurement and cheap corrective re-blend
# ============================================================


def measure_seam_diff(image: Image.Image, sample_step: int = 2) -> tuple[float, float]:
    """Average and max L1 pixel difference between the leftmost and rightmost
    column of an ERP panorama -- i.e. how visible the 0/360 wrap seam is.

    For reference: HY-Pano's own cleanly-blended examples measure ~4-10 here.
    Values above ~25-30 correspond to a visibly hard cut in a 360 viewer.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    left = arr[::sample_step, 0, :]
    right = arr[::sample_step, -1, :]
    diffs = np.abs(left - right).sum(axis=1)
    return float(diffs.mean()), float(diffs.max())


def reblend_final_edges(image: Image.Image, blend_width: int = 48) -> Image.Image:
    """Pull the leftmost and rightmost columns of an already-cropped panorama
    toward each other (via their shared per-row average), tapering back to
    the original pixels over `blend_width` columns on each side, without
    changing the image size.

    Unlike circular_blend_edges() (which blends a wider pre-crop render down
    to the final width, at generation time), this works on a finished image
    and is meant as a cheap corrective pass when measure_seam_diff() reports a
    bad seam and a full re-generation isn't wanted. Guaranteed to reduce the
    col[0]-vs-col[-1] diff at the boundary itself, since both columns are
    pulled toward the same per-row average at x=0.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float64).copy()
    w = arr.shape[1]
    bw = max(1, min(blend_width, w // 2))
    target = (arr[:, 0, :] + arr[:, -1, :]) / 2.0  # per-row average of the two edges
    for x in range(bw):
        t = 1.0 - x / bw  # 1.0 at the true edge, -> 0 moving inward
        arr[:, x, :] = arr[:, x, :] * (1 - t) + target * t
        arr[:, w - 1 - x, :] = arr[:, w - 1 - x, :] * (1 - t) + target * t
    return Image.fromarray(arr.astype(np.uint8))


def validate_and_fix_seam(
    image: Image.Image,
    *,
    seam_threshold: float = 25.0,
    reblend_width: int = 48,
) -> tuple[Image.Image, dict]:
    """Measure the seam and, if it's bad, apply a wider cross-fade as a cheap
    fix. Returns (possibly-reblended image, metrics dict).

    Does NOT retry generation with a new seed -- that needs to re-run the
    diffusion call and belongs in the caller (HunyuanPanoPipeline.forward),
    which has access to do so.
    """
    avg_before, max_before = measure_seam_diff(image)
    metrics: dict = {
        "seam_avg_before": avg_before,
        "seam_max_before": max_before,
        "reblended": False,
    }
    if avg_before > seam_threshold:
        image = reblend_final_edges(image, blend_width=reblend_width)
        avg_after, max_after = measure_seam_diff(image)
        metrics.update(
            reblended=True,
            seam_avg_after=avg_after,
            seam_max_after=max_after,
        )
    return image, metrics


# ============================================================
# 2 & 3. EXPERIMENTAL -- see module docstring for the strength-control caveat.
# ============================================================

POLE_CORRECTION_PROMPT = (
    "Perfectly straight, evenly spaced architectural grid lines, precise "
    "geometric alignment matching real-world vanishing point perspective, "
    "no warping, no waviness, sharp clean edges."
)

COLOR_REFINEMENT_PROMPT = (
    "Enhance sharpness and fine detail, correct color and exposure consistency "
    "across the entire image, preserve all existing structure, architecture, "
    "and objects exactly as shown, do not add, remove, or alter any content."
)


def _require_py360convert():
    try:
        import py360convert
        return py360convert
    except ImportError as exc:
        raise ImportError(
            "correct_poles() needs the 'py360convert' package (pip install "
            "py360convert) -- not installed in this environment."
        ) from exc


def correct_poles(
    pipe,
    image: Image.Image,
    *,
    negative_prompt: str,
    cap_fov: float = 90.0,
    out_size: int = 768,
    num_inference_steps: int = 30,
    true_cfg_scale: float = 7.5,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """EXPERIMENTAL. Reproject the zenith and nadir caps to rectilinear views,
    regenerate each with a straight-line-focused prompt, and feather them back
    into the panorama.

    `negative_prompt` should be the already fully-composed negative prompt
    (e.g. GENERAL_NEGATIVE_PROMPT + caller's extra text) -- this function does
    not build it itself.

    See the module docstring: because PanoDiffusionPipeline has no strength
    parameter, this can invent new detail in the cap region rather than just
    straightening existing lines. Inspect the result before trusting it.
    """
    import torch

    p360 = _require_py360convert()
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]

    result = arr.copy()
    for u_deg, v_deg in ((0.0, 90.0), (0.0, -90.0)):  # zenith, nadir
        cap = p360.e2p(
            arr, fov_deg=cap_fov, u_deg=u_deg, v_deg=v_deg,
            out_hw=(out_size, out_size),
        )
        cap_img = Image.fromarray(cap)
        corrected = pipe(
            image=cap_img,
            prompt=POLE_CORRECTION_PROMPT,
            negative_prompt=negative_prompt,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            height=out_size,
            width=out_size,
        ).images[0]

        back = p360.p2e(
            np.asarray(corrected), fov_deg=cap_fov, u_deg=u_deg, v_deg=v_deg,
            out_hw=(h, w),
        )
        # Feather the composite mask so stitching the cap back in doesn't add
        # a new hard edge at the cap boundary.
        mask = (back.sum(axis=2) > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=out_size * 0.03))
        mask_f = (np.asarray(mask_img, dtype=np.float64) / 255.0)[..., None]
        result = back.astype(np.float64) * mask_f + result.astype(np.float64) * (1 - mask_f)
        result = result.astype(np.uint8)

    return Image.fromarray(result)


def refine_colors(
    pipe,
    image: Image.Image,
    *,
    negative_prompt: str,
    num_inference_steps: int = 20,
    true_cfg_scale: float = 5.0,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """EXPERIMENTAL whole-panorama refinement pass. See module docstring for
    the strength-control caveat -- this is a full regeneration conditioned on
    the input image, not a gentle nudge.

    `negative_prompt` should be the already fully-composed negative prompt.
    """
    import torch

    w, h = image.size
    return pipe(
        image=image,
        prompt=COLOR_REFINEMENT_PROMPT,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        height=h,
        width=w,
    ).images[0]
