"""Command-line entry point for material-passport augmentation.

Examples
--------
Augment a single scene with the offline CV + rule methods::

    python -m material_passport.cli --scene sample_data/0

Augment all scenes using multiple material methods with consensus::

    python -m material_passport.cli --data-root sample_data \\
        --material-methods cv vlm --condition-methods rule vlm \\
        --material-methods-consensus priority \
        --condition-methods-consensus max_confidence --confidence-threshold 0.6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys

from . import condition as condition_mod, config
from .backends import ALL_METHODS
from .pipeline import process_scene
from .scene import list_scene_dirs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="material_passport",
        description="Augment ASE scenes with material type + condition.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", help="path to a single scene directory")
    src.add_argument("--data-root",
                     help="root containing numbered scene subdirectories")
    p.add_argument("--output-dir",
                   help="write all generated files (lang, JSON, viz, crops) "
                        "to this directory instead of the input scene dir; "
                        "the input scene's folder name is preserved as a "
                        "subdirectory. Example: --output-dir out/")
    p.add_argument("--material-methods", nargs="+", default=["cv"],
                   choices=ALL_METHODS,
                   help="one or more material backends to run")
    p.add_argument("--condition-methods", nargs="+", default=["rule"],
                   choices=condition_mod.ALL_CONDITION_METHODS,
                   help="one or more condition backends to run")
    p.add_argument("--material-methods-consensus", default=config.DEFAULT_CONSENSUS,
                   choices=["priority", "max_confidence", "vote"],
                   help="strategy for combining material backends "
                        "(with 'priority', --priority sets which method wins)")
    p.add_argument("--condition-methods-consensus", default=config.DEFAULT_CONSENSUS,
                   choices=["priority", "max_confidence", "vote"],
                   help="strategy for combining condition backends")
    p.add_argument("--confidence-threshold", type=float,
                   default=config.DEFAULT_CONFIDENCE_THRESHOLD,
                   help="threshold used by the 'priority' strategy")
    p.add_argument("--priority", nargs="+", default=config.METHOD_PRIORITY,
                   help="material method priority order for the 'priority' "
                        "strategy (first available + confident wins)")
    p.add_argument("--visualize", action="store_true",
                   help="write step-by-step images + text to "
                        "<scene>/material_viz/")
    p.add_argument("--viz-frames", type=int, default=config.VIZ_MAX_FRAMES,
                   help="number of association overlay frames to render")
    p.add_argument("--save-material-frames",
                   dest="save_material_frames", action="store_true",
                   help="also save the top-N full RGB frames used to estimate "
                        "material params per entity into "
                        "<scene>/material_frames/<entity_key>/. "
                        "OFF by default; use only when debugging or auditing "
                        "the frame selection.")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip scenes whose output lang file and JSON already "
                        "exist in the target output directory")
    p.add_argument("--num-workers", type=int, default=1,
                   help="number of scene-processing worker processes")
    p.add_argument("--num-gpus", type=int,
                   help="number of GPUs available on this node; workers are "
                        "assigned to GPUs round-robin")
    p.add_argument("--vlm-provider", choices=["auto", "ollama", "hf"],
                   help="override MP_VLM_PROVIDER for this run")
    p.add_argument("--vlm-model",
                   help="override MP_VLM_MODEL for this run, e.g. "
                        "google/gemma-4-E4B-it")
    p.add_argument("--quiet", action="store_true", help="suppress per-scene logs")
    return p


def _target_out_dir(scene_dir: str, output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    return os.path.join(output_dir, os.path.basename(scene_dir.rstrip("/")))


def _scene_output_complete(scene_dir: str, out_dir: str | None) -> bool:
    write_dir = out_dir if out_dir is not None else scene_dir
    return os.path.exists(os.path.join(write_dir, config.OUTPUT_LANG_FILE)) and \
        os.path.exists(os.path.join(write_dir, config.PASSPORT_JSON_FILE))


def _infer_num_gpus() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("NoDevFiles", "-1"):
        return len([x for x in visible.split(",") if x.strip()])
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        value = os.environ.get(name)
        if value and value.isdigit():
            return int(value)
    return 0


def _gpu_tokens(num_gpus: int) -> list[str]:
    if num_gpus <= 0:
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("NoDevFiles", "-1"):
        tokens = [x.strip() for x in visible.split(",") if x.strip()]
        if len(tokens) >= num_gpus:
            return tokens[:num_gpus]
    return [str(i) for i in range(num_gpus)]


def _run_one(scene_dir: str, args) -> dict:
    out_dir = _target_out_dir(scene_dir, args.output_dir)
    return process_scene(
        scene_dir,
        out_dir=out_dir,
        material_methods=args.material_methods,
        condition_methods=args.condition_methods,
        material_consensus=args.material_methods_consensus,
        condition_consensus=args.condition_methods_consensus,
        threshold=args.confidence_threshold,
        priority=args.priority,
        verbose=not args.quiet,
        visualize=args.visualize,
        save_material_frames=args.save_material_frames,
        viz_frames=args.viz_frames,
    )


def _worker(worker_id: int, gpu_token: str | None, tasks, results, args) -> None:
    if gpu_token is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token
        os.environ["MP_WORKER_GPU"] = gpu_token
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    while True:
        try:
            scene_dir = tasks.get()
        except (EOFError, KeyboardInterrupt):
            return
        if scene_dir is None:
            return
        try:
            results.put(("ok", _run_one(scene_dir, args)))
        except Exception as exc:  # pragma: no cover - exercised on HPC failures
            results.put(("error", scene_dir, worker_id, repr(exc)))


def _run_parallel(scenes: list[str], args, num_workers: int,
                  num_gpus: int) -> tuple[list[dict], list[tuple]]:
    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    tokens = _gpu_tokens(num_gpus)
    workers = []
    for worker_id in range(num_workers):
        gpu_token = tokens[worker_id % len(tokens)] if tokens else None
        p = ctx.Process(target=_worker,
                        args=(worker_id, gpu_token, tasks, results, args))
        p.start()
        workers.append(p)
    for scene_dir in scenes:
        tasks.put(scene_dir)
    for _ in workers:
        tasks.put(None)

    summaries = []
    errors = []
    received = 0
    while received < len(scenes):
        try:
            item = results.get(timeout=5)
        except queue.Empty:
            if any(p.is_alive() for p in workers):
                continue
            errors.append(("<worker-results>", "workers exited early",
                           f"{received}/{len(scenes)} results"))
            break
        received += 1
        if item[0] == "ok":
            summaries.append(item[1])
        else:
            errors.append(item[1:])
    for p in workers:
        p.join()
        if p.exitcode not in (0, None):
            errors.append(("<worker-exit>", p.pid, p.exitcode))
    return summaries, errors


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.vlm_provider:
        os.environ["MP_VLM_PROVIDER"] = args.vlm_provider
    if args.vlm_model:
        os.environ["MP_VLM_MODEL"] = args.vlm_model

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = list_scene_dirs(args.data_root)
        if not scenes:
            print(f"No scenes found under {args.data_root}", file=sys.stderr)
            return 1

    skipped = 0
    if args.skip_existing:
        pending = []
        for scene_dir in scenes:
            out_dir = _target_out_dir(scene_dir, args.output_dir)
            if _scene_output_complete(scene_dir, out_dir):
                skipped += 1
                if not args.quiet:
                    target = out_dir if out_dir is not None else scene_dir
                    print(f"[skip] {scene_dir} -> {target}")
            else:
                pending.append(scene_dir)
        scenes = pending

    num_workers = max(1, int(args.num_workers or 1))
    num_workers = min(num_workers, max(len(scenes), 1))
    num_gpus = args.num_gpus if args.num_gpus is not None else _infer_num_gpus()

    summaries: list[dict] = []
    errors: list[tuple] = []
    if len(scenes) > 1 and num_workers > 1:
        if not args.quiet:
            print(f"[workers] {num_workers} process(es), {num_gpus} GPU(s)")
        summaries, errors = _run_parallel(scenes, args, num_workers, num_gpus)
    else:
        if num_gpus > 0:
            tokens = _gpu_tokens(num_gpus)
            if tokens:
                os.environ["CUDA_VISIBLE_DEVICES"] = tokens[0]
                os.environ["MP_WORKER_GPU"] = tokens[0]
        for scene_dir in scenes:
            summaries.append(_run_one(scene_dir, args))

    if errors:
        for err in errors:
            print(f"[error] {err}", file=sys.stderr)
        return 1

    total_obs = sum(s["n_observed"] for s in summaries)
    total_ent = sum(s["n_entities"] for s in summaries)
    print(f"\nDone: {len(summaries)} processed, {skipped} skipped, "
          f"{total_obs}/{total_ent} entities observed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
