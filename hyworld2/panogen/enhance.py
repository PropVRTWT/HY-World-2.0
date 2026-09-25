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
            "This function needs the 'py360convert' package (pip install "
            "py360convert) -- not installed in this environment."
        ) from exc


# ============================================================
# 4. INPUT-SIDE: ERP conditioning (camera position / scale)
#
# The pipeline never told the model how wide the input photo is, so the model
# guessed what angular slice it occupies. Interiors are shot wide -- a frame
# showing 3 walls / 2 corners spans ~110-140 deg -- and if the model assumes
# ~70 deg it compresses the scene into half its true angular width, which reads
# as "the camera backed away from the room".
#
# project_input_to_erp() removes the guess: it places the photo into the ERP
# canvas at its true angular width using p2e(), so scale and position are fixed
# by geometry rather than inferred.
#
# CAVEAT: QwenImageEditPlus has no mask channel -- it cannot be told "these
# pixels are known, fill the rest". Feeding a canvas whose periphery is blank
# may work, or the model may treat the blank region as content to reproduce.
# `fill="blur"` (the default) exists for that reason: instead of hard black it
# extends the photo's edge colours outward and blurs them, so the unknown
# region reads as soft continuation rather than a black band. This whole path
# is OFF by default and unverified against a live model.
# ============================================================

# Horizontal field of view implied by how much of a room a single frame shows.
# Derived from counting visible wall planes / corners, which is something the
# uploader can answer reliably and a detector cannot (cluttered, low-contrast
# interiors defeat line-based vanishing-point estimation).
WALLS_TO_FOV = {
    1: 65.0,    # one wall flat-on, no corner in frame
    2: 105.0,   # two walls, one corner -- camera near a corner
    3: 130.0,   # three walls, two corners -- camera against the fourth wall
}


def fov_for_walls(walls: int) -> float:
    """Map a visible-wall count (1, 2 or 3) to an approximate horizontal FOV."""
    if walls not in WALLS_TO_FOV:
        raise ValueError(f"walls must be one of {sorted(WALLS_TO_FOV)}, got {walls!r}")
    return WALLS_TO_FOV[walls]


def perspective_to_equirect(
    arr: np.ndarray,
    *,
    fov_deg: float,
    out_hw: tuple[int, int],
    u_deg: float = 0.0,
    v_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a perspective image into an equirectangular canvas.

    py360convert ships e2p (equirect -> perspective) but NOT the inverse, so
    this is implemented directly: for every ERP pixel, build its view ray,
    rotate it into camera space, and pinhole-project it back into the source
    image. Inverse mapping (destination -> source) rather than forward, so the
    result has no holes.

    `fov_deg` is the HORIZONTAL field of view; the vertical extent follows from
    the source image's aspect ratio via the shared focal length, which is the
    correct pinhole behaviour.

    Returns (canvas, known-mask) where known-mask is True wherever a source
    pixel actually landed.
    """
    H, W = out_hw
    h_img, w_img = arr.shape[:2]

    lon = (np.arange(W) + 0.5) / W * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (np.arange(H) + 0.5) / H * np.pi
    lon, lat = np.meshgrid(lon, lat)

    # View ray in world space: +Z forward, +X right, +Y up.
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    # Rotate so the camera's optical axis points at (u_deg, v_deg).
    u, v = np.deg2rad(u_deg), np.deg2rad(v_deg)
    xr = x * np.cos(u) - z * np.sin(u)
    zt = x * np.sin(u) + z * np.cos(u)
    yr = y * np.cos(v) - zt * np.sin(v)
    zr = y * np.sin(v) + zt * np.cos(v)

    focal = (w_img / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    known = zr > 1e-6  # everything behind the camera is unseen
    zs = np.where(known, zr, 1.0)
    px = focal * xr / zs + (w_img - 1) / 2.0
    py = -focal * yr / zs + (h_img - 1) / 2.0
    known &= (px >= 0) & (px <= w_img - 1) & (py >= 0) & (py <= h_img - 1)

    # Bilinear sample (clamped so invalid coords stay in bounds; masked after).
    pxc = np.clip(px, 0, w_img - 1)
    pyc = np.clip(py, 0, h_img - 1)
    x0 = np.floor(pxc).astype(np.intp); x1 = np.minimum(x0 + 1, w_img - 1)
    y0 = np.floor(pyc).astype(np.intp); y1 = np.minimum(y0 + 1, h_img - 1)
    wx = (pxc - x0)[..., None]
    wy = (pyc - y0)[..., None]
    src = arr.astype(np.float64)
    out = (
        src[y0, x0] * (1 - wx) * (1 - wy)
        + src[y0, x1] * wx * (1 - wy)
        + src[y1, x0] * (1 - wx) * wy
        + src[y1, x1] * wx * wy
    )
    out[~known] = 0
    return out.astype(np.uint8), known


def _fill_unknown(erp: np.ndarray, known: np.ndarray, mode: str) -> np.ndarray:
    """Fill the region outside the projected photo.

    "black" leaves it at 0; "gray" uses mid-grey; "blur" propagates each row's
    outermost known pixel outward and blurs the result, so the model sees a
    soft continuation of the photo's tone rather than a hard edge.
    """
    if mode == "black":
        return erp
    out = erp.astype(np.float64).copy()
    if mode == "gray":
        out[~known] = 128.0
        return out.astype(np.uint8)
    if mode != "blur":
        raise ValueError(f"fill must be black|gray|blur, got {mode!r}")

    h, w = known.shape
    rows_with_content = np.flatnonzero(known.any(axis=1))
    if rows_with_content.size == 0:
        return erp
    # Horizontal pass: extend each covered row's outermost pixels sideways.
    for y in rows_with_content:
        cols = np.flatnonzero(known[y])
        lo, hi = cols[0], cols[-1]
        out[y, :lo, :] = out[y, lo, :]
        out[y, hi + 1:, :] = out[y, hi, :]
    # Vertical pass: a photo narrower than 180 deg vertically leaves whole rows
    # uncovered at the zenith and nadir. Left black they become dark ceilings
    # and floors in the output, so inherit from the nearest covered row.
    top, bot = rows_with_content[0], rows_with_content[-1]
    out[:top, :, :] = out[top, :, :]
    out[bot + 1:, :, :] = out[bot, :, :]
    blurred = np.asarray(
        Image.fromarray(out.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=24)),
        dtype=np.float64,
    )
    # keep the real photo pixels crisp; only the invented region is blurred
    blurred[known] = erp.astype(np.float64)[known]
    return blurred.astype(np.uint8)


def project_input_to_erp(
    image: Image.Image,
    *,
    fov_deg: float,
    out_hw: tuple[int, int],
    u_deg: float = 0.0,
    v_deg: float = 0.0,
    fill: str = "blur",
) -> tuple[Image.Image, float]:
    """Place a perspective photo into an ERP canvas at its true angular width.

    Args:
        image:   the input photo (perspective projection).
        fov_deg: its horizontal field of view. Use fov_for_walls() if you only
                 know how many walls are visible.
        out_hw:  (height, width) of the ERP canvas, e.g. (960, 1952).
        u_deg:   yaw placement; 0 centres the photo in the panorama.
        v_deg:   pitch placement; 0 puts it on the horizon.
        fill:    how to fill the region the photo does not cover --
                 "blur" (default), "gray" or "black".

    Returns:
        (ERP canvas as a PIL image, fraction of the canvas width the photo covers)
    """
    arr = np.asarray(image.convert("RGB"))
    erp, known = perspective_to_equirect(
        arr, fov_deg=fov_deg, out_hw=out_hw, u_deg=u_deg, v_deg=v_deg,
    )
    filled = _fill_unknown(erp, known, fill)
    coverage = known.any(axis=0).mean()
    return Image.fromarray(filled), float(coverage)


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

        # py360convert has no inverse of e2p, so use our own projection.
        back, back_known = perspective_to_equirect(
            np.asarray(corrected), fov_deg=cap_fov, out_hw=(h, w),
            u_deg=u_deg, v_deg=v_deg,
        )
        # Feather the composite mask so stitching the cap back in doesn't add
        # a new hard edge at the cap boundary.
        mask = back_known.astype(np.uint8) * 255
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
