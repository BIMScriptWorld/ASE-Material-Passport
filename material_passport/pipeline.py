"""Per-scene orchestration of the material-passport pipeline."""
from __future__ import annotations

import os

from . import backends, condition as condition_mod, config, consensus
from .association import observe_scene
from .camera import SceneCamera
from .scene import load_entities
from .viz import Visualizer
from .writer import write_scene_outputs
from .backends import vlm_backend


def _build_material_backends(methods, warn):
    out = []
    for m in methods:
        be = backends.get_material_backend(m)
        if be.available():
            out.append(be)
        else:
            warn(f"  [skip] material backend '{m}' unavailable")
    if not out:  # cv always works -> guarantee a result
        out.append(backends.get_material_backend("cv"))
        warn("  [info] falling back to material backend 'cv'")
    return out


def _build_condition_backends(methods, warn):
    out = []
    for m in methods:
        be = condition_mod.get_condition_backend(m)
        if be.available():
            out.append(be)
        else:
            warn(f"  [skip] condition backend '{m}' unavailable")
    if not out:
        out.append(condition_mod.get_condition_backend("rule"))
        warn("  [info] falling back to condition backend 'rule'")
    return out


def process_scene(scene_dir, material_methods, condition_methods,
                  out_dir=None,
                  material_consensus=config.DEFAULT_CONSENSUS,
                  condition_consensus=config.DEFAULT_CONSENSUS,
                  threshold=config.DEFAULT_CONFIDENCE_THRESHOLD,
                  priority=None, verbose=True, visualize=False,
                  save_material_frames=False,
                  viz_frames=config.VIZ_MAX_FRAMES):
    priority = priority or config.METHOD_PRIORITY
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"[scene] {scene_dir}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        log(f"[out]   {out_dir}")
    write_dir = out_dir if out_dir is not None else scene_dir

    entities, raw_lines = load_entities(scene_dir)
    camera = SceneCamera(scene_dir)

    viz = Visualizer(write_dir, enabled=visualize, max_frames=viz_frames,
                     log=log)
    viz.set_entities(entities)
    viz.floor_plan()
    viz.surface_points()

    cache_dir = os.path.join(write_dir, config.CACHE_DIRNAME)
    need_crops = backends.needs_crops(material_methods) or \
        "vlm" in condition_methods or visualize
    observations = observe_scene(scene_dir, entities, camera, need_crops,
                                 cache_dir=cache_dir,
                                 save_material_frames=save_material_frames,
                                 visualizer=viz)
    viz.association_frames()

    mat_backends = _build_material_backends(material_methods, log)
    cond_backends = _build_condition_backends(condition_methods, log)

    records: dict[int, dict] = {}
    for ob in observations:
        ent = ob.entity
        mat_preds = [b.predict(ob) for b in mat_backends]
        cond_preds = [b.predict(ob) for b in cond_backends]
        mat = consensus.decide(mat_preds, material_consensus, threshold,
                               priority)
        cond = consensus.decide(cond_preds, condition_consensus, threshold,
                                config.CONDITION_PRIORITY)
        ob._viz_material = mat.label
        ob._viz_condition = cond.label
        rec = {
            "id": ent.key,
            "class": ent.cls,
            "command": ent.command,
            "line_index": ent.line_index,
            "seg_id": ob.seg_id,
            "n_samples": ob.n_samples,
            "observed": ob.observed,
            "material": mat.label,
            "material_method": mat.method,
            "material_confidence": round(mat.confidence, 3),
            "material_components": mat.components,
            "material_candidates": [
                {"method": p.method, "label": p.label,
                 "confidence": round(p.confidence, 3)}
                for p in mat.candidates],
            "condition": cond.label,
            "condition_method": cond.method,
            "condition_confidence": round(cond.confidence, 3),
            "condition_candidates": [
                {"method": p.method, "label": p.label,
                 "confidence": round(p.confidence, 3)}
                for p in cond.candidates],
            "features": ob.features,
            "crops": [os.path.relpath(c, write_dir) for c in ob.crop_paths],
            "frames": [os.path.relpath(f, write_dir)
                       for f in ob.frame_paths],
        }
        if ent.line_index >= 0:
            records[ent.line_index] = rec

    lang_path, json_path = write_scene_outputs(write_dir, raw_lines, records)

    viz.crops(observations)
    viz.materials_floor_plan(records)
    viz.decisions(records)
    viz.finalize()
    vlm_backend.clear_result_cache()

    n_obs = sum(1 for r in records.values() if r["observed"])
    log(f"  entities={len(records)} observed={n_obs} -> {lang_path}")
    return {"scene_dir": scene_dir, "n_entities": len(records),
            "n_observed": n_obs, "lang_path": lang_path,
            "json_path": json_path,
            "viz_dir": viz.out_dir if visualize else None}
