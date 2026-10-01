"""Scene-language parsing and entity surface geometry.

Wraps the official ASE helpers (``read_language_file`` and
``language_to_bboxes``) and adds:
  * the raw source line index for each command (needed by the writer);
  * a set of world-space sample points on each entity's visible face.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from projectaria_tools.projects import ase

from . import config


@dataclass
class Entity:
    """A single make_wall / make_door / make_window command."""

    command: str            # "make_wall" | "make_door" | "make_window"
    cls: str                # "wall" | "door" | "window"
    identifier: int         # numeric id from the command
    line_index: int         # 0-based index of the source line
    center: np.ndarray      # (3,) world-space box center
    rotation: np.ndarray    # (3, 3) world-from-local rotation
    scale: np.ndarray       # (length/width, thickness, height)
    surface_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def key(self) -> str:
        return f"{self.cls}{self.identifier}"


def _surface_points(center: np.ndarray, rotation: np.ndarray,
                    scale: np.ndarray,
                    openings: Optional[list] = None) -> np.ndarray:
    """Sample a grid of points on the entity's main face (local y = 0 plane).

    ``openings`` is an optional list of ``(u0, w0, half_w, half_h)`` rectangles
    (in this entity's local frame) to exclude -- used to carve door/window
    cutouts out of a wall so its samples do not fall on coplanar glass/doors.
    """
    width, _thickness, height = float(scale[0]), float(scale[1]), float(scale[2])

    def _n(extent: float) -> int:
        n = int(round(abs(extent) / config.SAMPLE_SPACING_M)) + 1
        return int(np.clip(n, config.MIN_SAMPLES_PER_AXIS,
                           config.MAX_SAMPLES_PER_AXIS))

    us = np.linspace(-0.5 * width, 0.5 * width, _n(width))
    ws = np.linspace(-0.5 * height, 0.5 * height, _n(height))
    uu, ww = np.meshgrid(us, ws)
    u = uu.ravel()
    w = ww.ravel()
    keep = np.ones(u.shape[0], dtype=bool)
    for u0, w0, hw, hh in (openings or []):
        keep &= ~((np.abs(u - u0) <= hw) & (np.abs(w - w0) <= hh))
    if keep.sum() < config.MIN_SAMPLES_PER_AXIS ** 2:
        keep = np.ones(u.shape[0], dtype=bool)  # opening covers ~all: keep grid
    local = np.stack([u[keep], np.zeros(int(keep.sum())), w[keep]], axis=1)
    return center[None, :] + local @ rotation.T


def load_entities(scene_dir: str) -> tuple[list[Entity], list[str]]:
    """Parse a scene directory.

    Returns ``(entities, raw_lines)`` where ``raw_lines`` are the original
    ``ase_scene_language.txt`` lines (without trailing newlines) and
    ``entities`` carry geometry + the source line index.
    """
    lang_path = os.path.join(scene_dir, "ase_scene_language.txt")
    with open(lang_path, "r") as f:
        raw_lines = [ln.rstrip("\n") for ln in f.readlines()]

    # Map (command, id) -> source line index.
    line_lookup: dict[tuple[str, int], int] = {}
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split(", ")
        command = parts[0]
        params = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v
        if "id" in params:
            try:
                line_lookup[(command, int(float(params["id"])))] = idx
            except ValueError:
                continue

    parsed = ase.readers.read_language_file(lang_path)
    boxes = ase.interpreter.language_to_bboxes(parsed)

    entities: list[Entity] = []
    for box in boxes:
        cls = box["class"]
        command = box["cmd"]
        identifier = int(box["id"][len(cls):])
        center = np.asarray(box["center"], dtype=float)
        rotation = np.asarray(box["rotation"], dtype=float)
        scale = np.asarray(box["scale"], dtype=float)
        line_index = line_lookup.get((command, identifier), -1)
        ent = Entity(
            command=command,
            cls=cls,
            identifier=identifier,
            line_index=line_index,
            center=center,
            rotation=rotation,
            scale=scale,
        )
        entities.append(ent)

    # Map wall id -> the wall Entity, so we can carve openings out of walls.
    walls = {e.identifier: e for e in entities if e.cls == "wall"}
    wall_openings: dict[int, list] = {wid: [] for wid in walls}
    for e in entities:
        if e.cls not in ("door", "window"):
            continue
        for wid, wall in walls.items():
            # Express the opening center in the wall's local (u, w) frame.
            delta = e.center - wall.center
            local = wall.rotation.T @ delta
            u0, w0 = float(local[0]), float(local[2])
            half_len = 0.5 * float(wall.scale[0])
            half_h = 0.5 * float(wall.scale[2])
            # Opening belongs to this wall if it lies within the wall face and
            # is coplanar (small out-of-plane offset).
            if abs(u0) <= half_len + 0.5 and abs(w0) <= half_h + 0.5 \
                    and abs(float(local[1])) <= 0.3:
                wall_openings[wid].append(
                    (u0, w0, 0.5 * float(e.scale[0]), 0.5 * float(e.scale[2])))

    for e in entities:
        openings = wall_openings.get(e.identifier) if e.cls == "wall" else None
        e.surface_points = _surface_points(
            e.center, e.rotation, e.scale, openings)
    return entities, raw_lines


def list_scene_dirs(data_root: str) -> list[str]:
    """Return numerically sorted scene subdirectories under ``data_root``."""
    out = []
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "ase_scene_language.txt")):
            out.append(path)
    out.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return out
