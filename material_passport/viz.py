"""Step-by-step visualization of the material-passport pipeline.

When enabled (``--visualize``) this writes annotated images and text into
``<scene>/material_viz/`` so each stage can be inspected:

  00_steps.md                running text log of every stage (optional)
  01_floorplan.png           top-down plan of walls/doors/windows with ids
  02_surface_points.png      3D surface samples per entity (shows carved openings)
  03_association/frame_*.png  RGB + segmentation with validated projected pixels
  04_crops/<entity>.png       best RGB crops used for material inference
  05_materials_floorplan.png  plan recoloured by inferred material + condition
  06_decisions.png / .txt     per-entity material/condition decision table

The class is a safe no-op when ``enabled`` is False, so the pipeline can always
construct one unconditionally.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import config


def _get_cmap(name: str):
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        import matplotlib.cm as cm
        return cm.get_cmap(name)


def _entity_endpoints_xy(entity) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the two floor endpoints of a wall/opening from its box."""
    direction = entity.rotation[:, 0][:2]
    half = 0.5 * float(entity.scale[0])
    center_xy = entity.center[:2]
    return center_xy - direction * half, center_xy + direction * half


class Visualizer:
    def __init__(self, scene_dir: str, enabled: bool,
                 max_frames: int = config.VIZ_MAX_FRAMES,
                 log=print):
        self.enabled = enabled
        self.scene_dir = scene_dir
        self.max_frames = max_frames
        self._log = log
        self.out_dir = os.path.join(scene_dir, config.VIZ_DIRNAME)
        self.entities = []
        self.colors: dict[str, tuple] = {}
        self.frame_ids: set[int] = set()
        self.rgb_paths: list[str] = []
        self.inst_paths: list[str] = []
        # frame_idx -> {entity_key: (px array, py array)}
        self._records: dict[int, dict] = {}
        self._steps: list[str] = []
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "03_association"),
                        exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "04_crops"), exist_ok=True)
            import matplotlib
            matplotlib.use("Agg")

    # -- bookkeeping -----------------------------------------------------
    def step(self, msg: str):
        if not self.enabled:
            return
        self._steps.append(msg)
        self._log(f"  [viz] {msg}")

    def set_entities(self, entities):
        if not self.enabled:
            return
        self.entities = entities
        cmap = _get_cmap("tab20")
        for i, e in enumerate(entities):
            self.colors[e.key] = cmap(i % 20)
        self.step(f"parsed {len(entities)} entities "
                  f"({_count_classes(entities)})")

    def set_frame_sources(self, rgb_paths, inst_paths, n_frames):
        if not self.enabled:
            return
        self.rgb_paths = rgb_paths
        self.inst_paths = inst_paths
        if n_frames <= 0:
            return
        k = min(self.max_frames, n_frames)
        self.frame_ids = set(int(x) for x in
                             np.linspace(0, n_frames - 1, k).round())
        self.step(f"selected {len(self.frame_ids)} frames for association "
                  f"overlays: {sorted(self.frame_ids)}")

    def wants_frame(self, frame_idx: int) -> bool:
        return self.enabled and frame_idx in self.frame_ids

    def record_pixels(self, frame_idx, entity_key, px, py):
        if not self.enabled:
            return
        self._records.setdefault(frame_idx, {})[entity_key] = (
            np.asarray(px), np.asarray(py))

    # -- renders ---------------------------------------------------------
    def floor_plan(self):
        if not self.enabled:
            return
        self._draw_plan(os.path.join(self.out_dir, "01_floorplan.png"),
                        title="Scene floor plan (ASE entities)")
        self.step("rendered 01_floorplan.png")

    def surface_points(self):
        if not self.enabled:
            return
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        for e in self.entities:
            pts = e.surface_points
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], s=2,
                           color=self.colors[e.key], label=None)
        self._overlay_walls(ax)
        ax.set_aspect("equal")
        ax.set_title("Surface sample points (top-down; openings carved out)")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.tight_layout(); fig.savefig(
            os.path.join(self.out_dir, "02_surface_points.png"), dpi=130)
        plt.close(fig)
        self.step("rendered 02_surface_points.png")

    def association_frames(self):
        if not self.enabled or not self._records:
            return
        import matplotlib.pyplot as plt
        from PIL import Image
        path_by_idx = {i: p for i, p in enumerate(self.rgb_paths)}
        inst_by_idx = {i: p for i, p in enumerate(self.inst_paths)}
        n = 0
        for fi in sorted(self._records):
            rgb = np.asarray(Image.open(path_by_idx[fi]).convert("RGB"))
            inst = np.asarray(Image.open(inst_by_idx[fi]), dtype=np.int32)
            fig, axes = plt.subplots(1, 2, figsize=(14, 8))
            axes[0].imshow(rgb); axes[0].set_title(f"RGB + projected pixels "
                                                   f"(frame {fi})")
            axes[1].imshow(_colorize_instances(inst))
            axes[1].set_title("Instance segmentation + projected pixels")
            for key, (px, py) in self._records[fi].items():
                c = self.colors.get(key, (1, 0, 0, 1))
                for ax in axes:
                    ax.scatter(px, py, s=4, color=c, edgecolors="none")
            handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                  color=self.colors.get(k, "r"), label=k)
                       for k in self._records[fi]]
            axes[0].legend(handles=handles, fontsize=7, loc="upper right",
                           framealpha=0.6)
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "03_association",
                                     f"frame_{fi:07d}.png"), dpi=110)
            plt.close(fig)
            n += 1
        self.step(f"rendered {n} association overlay(s) in 03_association/")

    def crops(self, observations):
        if not self.enabled:
            return
        from PIL import Image, ImageDraw
        rec_by_key = {}
        for ob in observations:
            rec_by_key[ob.entity.key] = ob
        made = 0
        for ob in observations:
            paths = ob.crop_paths[: config.VIZ_MAX_CROPS]
            if not paths:
                continue
            imgs = [Image.open(p).convert("RGB") for p in paths]
            h = max(im.height for im in imgs)
            imgs = [im.resize((int(im.width * h / im.height), h))
                    for im in imgs]
            gap = 6
            total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
            bar = 22
            canvas = Image.new("RGB", (max(total_w, 160), h + bar),
                               (30, 30, 30))
            x = 0
            for im in imgs:
                canvas.paste(im, (x, bar)); x += im.width + gap
            d = ImageDraw.Draw(canvas)
            mat = getattr(ob, "_viz_material", "?")
            cond = getattr(ob, "_viz_condition", "?")
            d.text((4, 5), f"{ob.entity.key}  material={mat}  condition={cond}",
                   fill=(240, 240, 240))
            canvas.save(os.path.join(self.out_dir, "04_crops",
                                     f"{ob.entity.key}.png"))
            made += 1
        self.step(f"rendered {made} entity crop montage(s) in 04_crops/")

    def materials_floor_plan(self, records: dict):
        if not self.enabled:
            return
        self._draw_plan(
            os.path.join(self.out_dir, "05_materials_floorplan.png"),
            title="Inferred materials & condition", records=records,
            color_by_material=True)
        self.step("rendered 05_materials_floorplan.png")

    def decisions(self, records: dict):
        if not self.enabled:
            return
        rows = sorted(records.values(), key=lambda r: r["line_index"])
        # Text table
        lines = [f"{'id':>11} {'class':<7} {'seg':>4} {'samples':>8} "
                 f"{'material':<25} {'m_meth':<6} {'m_cf':>5} "
                 f"{'condition':<10} {'c_meth':<6} {'c_cf':>5}"]
        for r in rows:
            mat = r['material']
            if mat == "composite":
                mat = f"composite ({'/'.join(r['material_components'])})"
            lines.append(
                f"{r['id']:>11} {r['class']:<7} {str(r['seg_id']):>4} "
                f"{r['n_samples']:>8} {str(mat):<25} "
                f"{r['material_method']:<6} {r['material_confidence']:>5.2f} "
                f"{str(r['condition']):<10} {r['condition_method']:<6} "
                f"{r['condition_confidence']:>5.2f}")
        txt = "\n".join(lines)
        with open(os.path.join(self.out_dir, "06_decisions.txt"), "w") as f:
            f.write(txt + "\n")
        # # Figure table
        # import matplotlib.pyplot as plt
        # col_labels = ["id", "class", "seg", "material", "m_cf",
        #               "condition", "c_cf", "material candidates"]
        # cell_text = []
        # for r in rows:
        #     cands = " ".join(f"{c['method']}:{c['label']}={c['confidence']:.2f}"
        #                      for c in r["material_candidates"])
        #     cell_text.append([
        #         r["id"], r["class"], str(r["seg_id"]), str(r["material"]),
        #         f"{r['material_confidence']:.2f}", str(r["condition"]),
        #         f"{r['condition_confidence']:.2f}", cands])
        # fig_h = max(2.0, 0.45 * (len(cell_text) + 1))
        # fig, ax = plt.subplots(figsize=(14, fig_h))
        # ax.axis("off")
        # tbl = ax.table(cellText=cell_text, colLabels=col_labels,
        #                loc="center", cellLoc="left")
        # tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.3)
        # ax.set_title("Per-entity material & condition decisions")
        # fig.tight_layout()
        # fig.savefig(os.path.join(self.out_dir, "06_decisions.png"), dpi=130)
        # plt.close(fig)
        self.step("rendered 06_decisions.txt")

    def finalize(self):
        if not self.enabled:
            return
        md = ["# Material passport visualization",
              f"Scene: `{os.path.basename(self.scene_dir)}`", "",
              "## Pipeline steps"]
        for i, s in enumerate(self._steps, 1):
            md.append(f"{i}. {s}")
        md += ["", "## Artifacts",
               "- `01_floorplan.png` - parsed ASE geometry (top-down)",
               "- `02_surface_points.png` - per-entity surface samples",
               "- `03_association/` - projected pixels on RGB + segmentation",
               "- `04_crops/` - RGB crops used for material inference",
               "- `05_materials_floorplan.png` - inferred materials/condition",
               "- `06_decision.txt` - decision table"]
        # with open(os.path.join(self.out_dir, "00_steps.md"), "w") as f:
        #     f.write("\n".join(md) + "\n")
        self._log(f"  [viz] wrote visualization to {self.out_dir}")

    # -- helpers ---------------------------------------------------------
    def _overlay_walls(self, ax):
        for e in self.entities:
            if e.cls != "wall":
                continue
            a, b = _entity_endpoints_xy(e)
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.4", lw=1.0, zorder=1)

    def _draw_plan(self, out_path, title, records=None,
                   color_by_material=False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        mat_colors = {}
        if color_by_material and records:
            mats = sorted({(r["material"]) for r in records.values()
                           if r["material"]})
            cmap = _get_cmap("tab20")
            mat_colors = {m: cmap(i % 20) for i, m in enumerate(mats)}

        rec_by_line = records or {}
        line_of_key = {e.key: e.line_index for e in self.entities}

        for e in self.entities:
            rec = rec_by_line.get(line_of_key.get(e.key, -1))
            if e.cls == "wall":
                a, b = _entity_endpoints_xy(e)
                if color_by_material and rec:
                    col = mat_colors.get(rec["material"], "0.5")
                    lw = 5
                else:
                    col = self.colors.get(e.key, "0.3"); lw = 3
                ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw,
                        solid_capstyle="round")
                mx, my = (a + b) * 0.5
                if color_by_material and rec:
                    ax.annotate(f"{rec['material']}\n{rec['condition']}",
                                (mx, my), fontsize=6, ha="center",
                                color="black")
                else:
                    ax.annotate(e.key, (mx, my), fontsize=7, ha="center")
            else:
                a, b = _entity_endpoints_xy(e)
                marker = "s" if e.cls == "door" else "D"
                if color_by_material and rec:
                    col = mat_colors.get(rec["material"], "0.5")
                else:
                    col = self.colors.get(e.key, "0.3")
                ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=6,
                        solid_capstyle="butt")
                cx, cy = e.center[:2]
                ax.scatter([cx], [cy], marker=marker, s=60, color=col,
                           edgecolors="black", zorder=3)
                label = (f"{rec['material']}/{rec['condition']}"
                         if color_by_material and rec else e.key)
                ax.annotate(label, (cx, cy), fontsize=6, ha="center",
                            xytext=(0, 6), textcoords="offset points")

        if color_by_material and mat_colors:
            handles = [plt.Line2D([0], [0], color=c, lw=4, label=m)
                       for m, c in mat_colors.items()]
            ax.legend(handles=handles, fontsize=8, title="material",
                      loc="upper right", framealpha=0.7)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.tight_layout(); fig.savefig(out_path, dpi=130)
        plt.close(fig)


def _count_classes(entities) -> str:
    from collections import Counter
    c = Counter(e.cls for e in entities)
    return ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))


def _colorize_instances(inst: np.ndarray) -> np.ndarray:
    """Map a uint16 instance map to a stable pseudo-colour RGB image."""
    ids = inst.astype(np.int64)
    rng_r = (ids * 1103515245 + 12345) % 256
    rng_g = (ids * 2654435761 + 40503) % 256
    rng_b = (ids * 40503 + 12345) % 256
    out = np.stack([rng_r, rng_g, rng_b], axis=-1).astype(np.uint8)
    out[ids <= 1] = 0  # empty_space / background -> black
    return out
