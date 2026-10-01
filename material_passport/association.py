"""Associate ASE entities with image pixels and aggregate observations.

For each frame we project every entity's surface samples, validate them against
the rendered depth (occlusion test), then collect the RGB values and the
segmentation instance id seen at those pixels. This works without
``object_instances_to_classes.json`` (scenes 20-29) because association is
purely geometric.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from . import config, features
from .camera import SceneCamera
from .scene import Entity

_MAX_SAMPLES_PER_ENTITY = 40000


@dataclass
class EntityObservation:
    entity: Entity
    seg_id: int = -1
    n_samples: int = 0
    features: dict = field(default_factory=dict)
    crop_paths: list[str] = field(default_factory=list)
    # Full RGB frames (not just crops) the entity was best observed in; only
    # populated when --save-material-frames is enabled.
    frame_paths: list[str] = field(default_factory=list)
    # (count, frame_idx, (x0, y0, x1, y1)) for the best-observed frames.
    _frame_records: list = field(default_factory=list)

    @property
    def observed(self) -> bool:
        return self.n_samples >= config.MIN_ENTITY_SAMPLES


def _frame_paths(scene_dir: str, sub: str, prefix: str, ext: str) -> list[str]:
    d = os.path.join(scene_dir, sub)
    files = sorted(f for f in os.listdir(d) if f.startswith(prefix))
    return [os.path.join(d, f) for f in files]


def observe_scene(scene_dir: str, entities: list[Entity], camera: SceneCamera,
                  need_crops: bool, cache_dir: str = None,
                  save_material_frames: bool = False,
                  visualizer=None) -> list[EntityObservation]:
    rgb_paths = _frame_paths(scene_dir, "rgb", "vignette", ".jpg")
    depth_paths = _frame_paths(scene_dir, "depth", "depth", ".png")
    inst_paths = _frame_paths(scene_dir, "instances", "instance", ".png")
    n_frames = min(len(rgb_paths), len(depth_paths), len(inst_paths),
                   camera.num_frames)

    if visualizer is not None:
        visualizer.set_frame_sources(rgb_paths, inst_paths, n_frames)

    obs = [EntityObservation(entity=e) for e in entities]
    rgb_buffers = [[] for _ in entities]      # list of (k,3) uint8 arrays
    seg_votes = [Counter() for _ in entities]
    sample_counts = [0 for _ in entities]

    W, H = camera.width, camera.height

    for fi in range(n_frames):
        depth_img = np.asarray(Image.open(depth_paths[fi]), dtype=np.int32)
        inst_img = np.asarray(Image.open(inst_paths[fi]), dtype=np.int32)
        rgb_img = None  # lazily loaded only if a pixel is validated
        viz_frame = visualizer is not None and visualizer.wants_frame(fi)

        for ei, ent in enumerate(entities):
            uv, dist_m, _ = camera.project(ent.surface_points, fi)
            if uv.shape[0] == 0:
                continue
            px = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
            py = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
            depth_vals = depth_img[py, px]
            ray_mm = dist_m * 1000.0
            tol = np.maximum(config.DEPTH_TOL_MM,
                             config.DEPTH_TOL_FRAC * depth_vals)
            ok = (depth_vals > 0) & (np.abs(ray_mm - depth_vals) <= tol)
            if not ok.any():
                continue

            vpx, vpy = px[ok], py[ok]
            if viz_frame:
                visualizer.record_pixels(fi, ent.key, vpx, vpy)
            seg_here = inst_img[vpy, vpx]
            for s in seg_here:
                if s > 1:  # skip 0 empty_space / 1 background
                    seg_votes[ei][int(s)] += 1

            if rgb_img is None:
                rgb_img = np.asarray(Image.open(rgb_paths[fi]).convert("RGB"))
            samp = rgb_img[vpy, vpx]
            if sample_counts[ei] < _MAX_SAMPLES_PER_ENTITY:
                rgb_buffers[ei].append(samp)
                sample_counts[ei] += samp.shape[0]

            x0, y0 = int(vpx.min()), int(vpy.min())
            x1, y1 = int(vpx.max()) + 1, int(vpy.max()) + 1
            obs[ei]._frame_records.append((int(ok.sum()), fi, (x0, y0, x1, y1)))

    cache_dir = cache_dir or os.path.join(scene_dir, config.CACHE_DIRNAME)
    for ei, ob in enumerate(obs):
        if sample_counts[ei] == 0:
            continue
        rgb_all = np.concatenate(rgb_buffers[ei], axis=0)
        ob.n_samples = int(rgb_all.shape[0])
        ob.features = features.compute_color_features(rgb_all)
        if seg_votes[ei]:
            ob.seg_id = int(seg_votes[ei].most_common(1)[0][0])
        if need_crops and ob.observed:
            ob.crop_paths = _extract_crops(
                ob, rgb_paths, cache_dir, ent_key=ob.entity.key)
            ob.features["texture_laplacian"] = _crop_texture(ob.crop_paths)
        if save_material_frames and ob.observed:
            ob.frame_paths = _extract_material_frames(
                ob, rgb_paths, cache_dir, ent_key=ob.entity.key)
    return obs


def _extract_crops(ob: EntityObservation, rgb_paths: list[str],
                   cache_dir: str, ent_key: str) -> list[str]:
    os.makedirs(cache_dir, exist_ok=True)
    best = sorted(ob._frame_records, key=lambda r: r[0], reverse=True)
    paths: list[str] = []
    for count, fi, (x0, y0, x1, y1) in best[: config.N_CROP_FRAMES]:
        pad = config.CROP_PADDING_PX
        rgb_img = Image.open(rgb_paths[fi]).convert("RGB")
        W, Himg = rgb_img.size
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(W, x1 + pad), min(Himg, y1 + pad)
        if (cx1 - cx0) < config.MIN_CROP_SIZE_PX or \
                (cy1 - cy0) < config.MIN_CROP_SIZE_PX:
            continue
        crop = rgb_img.crop((cx0, cy0, cx1, cy1))
        out = os.path.join(cache_dir, f"{ent_key}_f{fi:07d}.png")
        crop.save(out)
        paths.append(out)
    return paths


def _extract_material_frames(ob: EntityObservation, rgb_paths: list[str],
                             cache_dir: str, ent_key: str) -> list[str]:
    """Save the top-N best-observed full RGB frames for this entity.

    Frames are written to ``<scene_or_output>/<MATERIAL_FRAMES_DIRNAME>/<key>/``
    (i.e. a sibling of ``material_cache``) so they don't pollute the cache
    directory used by the dms / vlm backends.
    """
    parent = os.path.dirname(os.path.abspath(cache_dir))
    out_dir = os.path.join(parent, config.MATERIAL_FRAMES_DIRNAME, ent_key)
    os.makedirs(out_dir, exist_ok=True)

    best = sorted(ob._frame_records, key=lambda r: r[0], reverse=True)
    saved: list[str] = []
    seen_frames: set[int] = set()
    for count, fi, _bbox in best:
        if fi in seen_frames:
            continue
        seen_frames.add(fi)
        out = os.path.join(out_dir, f"frame_f{fi:07d}.png")
        Image.open(rgb_paths[fi]).convert("RGB").save(out)
        saved.append(out)
        if len(saved) >= config.MATERIAL_FRAMES_PER_ENTITY:
            break
    return saved


def _crop_texture(crop_paths: list[str]) -> float:
    vals = []
    for p in crop_paths:
        img = np.asarray(Image.open(p).convert("L"))
        vals.append(features.laplacian_variance(img))
    return float(np.mean(vals)) if vals else 0.0
