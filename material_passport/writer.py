"""Write the augmented scene-language file and the JSON passport cache."""
from __future__ import annotations

import json
import os

from . import config


def _format_material(material: str, components) -> str:
    """Render a material value, expanding composite with its constituents.

    Constituents are slash-separated (not comma) so the augmented line stays a
    valid comma-delimited ``key=value`` record, e.g.
    ``material_type=composite (wood/metal)``.
    """
    if material == "composite" and components:
        return f"composite ({'/'.join(components)})"
    return material


def _append_params(line: str, material: str, condition: str) -> str:
    line = line.rstrip()
    return f"{line}, material_type={material}, condition={condition}"


def write_scene_outputs(scene_dir: str, raw_lines: list[str],
                        records: dict) -> tuple[str, str]:
    """Emit ``ase_scene_material_lang.txt`` and ``material_passport.json``.

    ``records`` maps source line index -> {"material", "condition", ...}.
    Every make_wall/make_door/make_window line gets both params appended
    (unknown entities receive vocabulary fallbacks).
    """
    out_lines: list[str] = []
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if stripped.startswith(("make_wall", "make_door", "make_window")):
            rec = records.get(idx)
            if rec is None:
                cls = stripped.split(",", 1)[0][len("make_"):]
                material = config.FALLBACK_MATERIAL.get(cls, "unknown")
                condition = config.FALLBACK_CONDITION
            else:
                material = _format_material(
                    rec["material"] or config.FALLBACK_MATERIAL.get(
                        rec["class"], "unknown"),
                    rec.get("material_components"))
                condition = rec["condition"] or config.FALLBACK_CONDITION
            out_lines.append(_append_params(line, material, condition))
        else:
            out_lines.append(line.rstrip())

    lang_path = os.path.join(scene_dir, config.OUTPUT_LANG_FILE)
    with open(lang_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    json_path = os.path.join(scene_dir, config.PASSPORT_JSON_FILE)
    with open(json_path, "w") as f:
        json.dump({"scene_dir": os.path.basename(scene_dir),
                   "entities": list(records.values())}, f, indent=2)
    return lang_path, json_path
