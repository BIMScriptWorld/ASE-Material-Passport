"""Static configuration: closed material vocabulary, conditions, defaults."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Closed material vocabulary (per ASE class). Backends MUST map into these.
# ---------------------------------------------------------------------------
MATERIAL_VOCAB = {
    "wall": [
        "painted_plaster",
        "wallpaper",
        "brick",
        "concrete",
        "wood_paneling",
        "tile",
        "stone",
        "wood",
        "metal",
        "glass",
        "composite",
    ],
    "door": [
        "wood",
        "metal",
        "glass",
        "composite",
    ],
    # For windows we report the frame material; glazing is always glass.
    "window": [
        "wood",
        "aluminum",
        "pvc",
    ],
}

# Condition vocabulary (ordered best -> worst).
CONDITION_VOCAB = ["new", "good", "worn", "damaged"]

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------
# Surface sampling grid resolution (points along the longer / shorter axis).
SAMPLE_SPACING_M = 0.12  # ~12 cm between surface sample points
MIN_SAMPLES_PER_AXIS = 4
MAX_SAMPLES_PER_AXIS = 60

# Depth occlusion tolerance: a projected surface point counts as "seen" if the
# rendered depth is within max(DEPTH_TOL_MM, DEPTH_TOL_FRAC * depth) of the
# point's distance along the camera ray.
DEPTH_TOL_MM = 60.0
DEPTH_TOL_FRAC = 0.06

# Minimum validated pixel samples for an entity to be considered observed.
MIN_ENTITY_SAMPLES = 25

# Number of best frames (by validated pixel count) to keep crops for.
N_CROP_FRAMES = 4
CROP_PADDING_PX = 8
MIN_CROP_SIZE_PX = 12

# Consensus
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_CONSENSUS = "priority"  # priority | max_confidence | vote
# Method priority order used by the "priority" consensus strategy.
METHOD_PRIORITY = ["vlm", "dms", "cv"]
# Condition method priority order (used by the "priority" strategy).
CONDITION_PRIORITY = ["vlm", "rule"]

# Fallbacks used when nothing can be inferred.
FALLBACK_MATERIAL = {
    "wall": "painted_plaster",
    "door": "wood",
    "window": "aluminum",
}
FALLBACK_CONDITION = "good"

# Output / cache file names (written inside each scene directory).
OUTPUT_LANG_FILE = "ase_scene_material_lang.txt"
PASSPORT_JSON_FILE = "material_passport.json"
CACHE_DIRNAME = "material_cache"
MATERIAL_FRAMES_DIRNAME = "material_frames"
MATERIAL_FRAMES_PER_ENTITY = 4   # top-N best-observed frames to save per entity

# Visualization
VIZ_DIRNAME = "material_viz"
VIZ_MAX_FRAMES = 6        # how many association overlay frames to render
VIZ_MAX_CROPS = 4         # crops per entity in the montage
