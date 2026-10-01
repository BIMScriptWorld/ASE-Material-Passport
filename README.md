# Material Passport Augmentation for ASE Scenes

Augment [Aria Synthetic Environments (ASE)](https://www.projectaria.com/datasets/ase/)
scenes with a **material passport**: for every wall, door, and window this
pipeline infers a **material type** (from a fixed vocabulary) and a **condition**
(`new` / `good` / `worn` / `damaged`), and writes the result back into an
augmented scene-language file plus a JSON cache.

It works by projecting each structural entity's surface into the posed RGB
frames, validating visibility against the depth + instance renders, collecting
texture crops, and classifying them with one or more pluggable backends.

---

## Dataset: ASE with Material Passport

We ran this pipeline over the ASE training scenes and release the results on
Hugging Face:

**🤗 [huggingface.co/datasets/prakashknaikade/ASE-Material-Passport](https://huggingface.co/datasets/prakashknaikade/ASE-Material-Passport)**

| Archive | Contents per scene | Use it for |
| --- | --- | --- |
| `ASE_with_Material_Passport_Slim.tar.gz` | `ase_scene_material_lang.txt`, `material_passport.json` | Training / evaluation on the material + condition labels |
| `ASE_with_Material_Passport_Full.tar.gz` | everything in Slim, plus `material_cache/` (texture crops), `material_frames/` and `material_viz/` where generated | Auditing the predictions, re-running consensus, visual inspection |

Both archives extract to one folder per ASE scene id (`<scene_id>/…`), matching
the scene ids of the original ASE release. The labels were produced with:

```
--material-methods dms vlm cv --condition-methods vlm rule
--material-methods-consensus priority --condition-methods-consensus priority
--vlm-provider hf --vlm-model google/gemma-4-E4B-it
```

Download, e.g. the slim archive:

```bash
pip install -U huggingface_hub
hf download prakashknaikade/ASE-Material-Passport \
  ASE_with_Material_Passport_Slim.tar.gz --repo-type dataset --local-dir .
tar -xzf ASE_with_Material_Passport_Slim.tar.gz
```

See [`sample_outputs/`](sample_outputs/) for a few example scenes, and
[section 1](#1-what-you-get-per-scene) for the file formats.

> **Labels are automatic pseudo-labels.** Materials and conditions are
> predicted by vision models and heuristics from synthetic renders. They have
> not been verified by humans, so treat them as noisy. Every entity's
> per-backend candidates and confidences are kept in `material_passport.json`,
> so you can filter or re-weight them.

### Terms of use

The released data is derived from the
[Aria Synthetic Environments dataset](https://www.projectaria.com/datasets/ase/)
and is provided **for non-commercial research purposes only**. All use of it,
including downstream use, is governed by the
[Aria Synthetic Environments Dataset License Agreement](https://www.projectaria.com/datasets/ase/license/).
By downloading it you agree to those terms. Any publication that uses it must
include the attribution required by that license:
*[Aria Synthetic Environments Dataset, Meta Reality Labs-R]*.

The MIT license in this repository covers the **code only**, not the data.

---

## 1. What you get per scene

Running the pipeline on a scene directory writes (inside that directory):

| File | Description |
| --- | --- |
| `ase_scene_material_lang.txt` | The original scene language with `material_type=…, condition=…` appended to every `make_wall` / `make_door` / `make_window` line. |
| `material_passport.json` | Per-entity record: matched segmentation id, sampled features, every backend's candidate prediction, and the final decision. |
| `material_cache/` | Texture crops used by the `vlm` / `dms` backends (only created when needed). |
| `material_frames/` | Full RGB frames used for material estimation audits (only with `--save-material-frames`). |
| `material_viz/` | Step-by-step images + text (only with `--visualize`). |

**Composite materials** are reported with their constituents, e.g.
`material_type=composite (wood/metal)` in the language file, and as
`"material": "composite", "material_components": ["wood", "metal"]` in the JSON.

---

## 2. Project layout

```
material_passport/        # the Python package
  cli.py                  # command-line entry point
  pipeline.py             # per-scene orchestration
  scene.py                # ASE language -> entities (+ surface sampling)
  camera.py               # ASE calibration + trajectory -> projection
  association.py          # project entities into frames, match segmentation
  features.py             # colour / texture features from sampled pixels
  consensus.py            # combine multiple backend predictions
  condition.py            # condition (wear) backends
  writer.py               # write augmented lang file + JSON passport
  viz.py                  # optional step visualisations
  backends/
    cv_backend.py         # offline heuristic (always available)
    vlm_backend.py        # Ollama or HF Transformers vision model
    dms_backend.py        # HF SigLIP MINC material classifier
sample_outputs/           # example pipeline outputs for a few ASE scenes
sbatch_material_ase_lang.sh  # SLURM job for multi-GPU runs over a dataset root
environment.yml           # conda env (core + ASE tooling)
requirements.txt          # pip alternative
sample_data/              # (not tracked) put your local ASE scenes here
ASE_tutorial/             # Meta's ASE downloaders + tutorial notebook (see 3.2)
  aria_synthetic_environments_downloader.py           # full scenes (what the pipeline needs)
  aria_synthetic_environments_downloader_just_req.py  # scene language + metadata only
  tutorial/                                           # ASE tutorial notebook + helpers
```

Each scene directory is expected to contain `ase_scene_language.txt`,
`trajectory.csv`, and `rgb/`, `depth/`, `instances/` frame folders (the standard
ASE layout).

---

## 3. Setup

### 3.1 Base environment (required)

```bash
conda env create -f environment.yml
conda activate material_passport
```

This installs `projectaria-tools` (ASE calibration / readers / interpreter),
`numpy`, `Pillow`, and `matplotlib`. With just this you can run the offline
`cv` material backend and the `rule` condition backend.

> Prefer pip? `pip install -r requirements.txt` into any Python 3.11 env.

### 3.2 Download ASE scenes

`ASE_tutorial/` contains Meta's ASE downloader scripts and tutorial notebook,
copied from [projectaria_tools](https://github.com/facebookresearch/projectaria_tools)
(Apache-2.0). The pipeline does not import them; they are here to help you
fetch ASE scenes.

1. Go to the [ASE dataset page](https://www.projectaria.com/datasets/ase/),
   accept the license, and download the CDN url file
   (`aria_synthetic_environments_dataset_download_urls.json`).
2. Download the scenes you want (needs `tqdm`: `pip install tqdm`):

   ```bash
   python ASE_tutorial/aria_synthetic_environments_downloader.py \
     --set train \
     --scene-ids 0-9 \
     --cdn-file aria_synthetic_environments_dataset_download_urls.json \
     --output-dir sample_data \
     --unzip True
   ```

   `--scene-ids` takes comma-separated ids and ranges, e.g. `0,5,10-19`. Scenes
   are downloaded in chunks of 10, so request whole chunks where you can.

`aria_synthetic_environments_downloader_just_req.py` takes the same arguments
but extracts only `ase_scene_language.txt`, `object_instances_to_classes.json`
and `semidense_points.csv.gz`. That is enough to pair a scene with its
released `material_passport.json`, but **not** enough to run this pipeline,
which also needs `trajectory.csv` and the `rgb/`, `depth/`, `instances/` frames.

> **Note:** both downloaders disable HTTPS certificate verification (as in
> Meta's original scripts); every downloaded chunk is still checked against
> the SHA-1 in the CDN url file before it is extracted.

The tutorial notebook (`ASE_tutorial/tutorial/ase_tutorial_notebook.ipynb`)
shows how to read and visualise an ASE scene with `projectaria_tools`.

### 3.3 Optional: `dms` backend (HuggingFace MINC classifier)

The `dms` backend runs the pretrained SigLIP material classifier
[`prithivMLmods/Minc-Materials-23`](https://huggingface.co/prithivMLmods/Minc-Materials-23).
It needs `transformers` and a `torch` build that matches your GPU/CUDA.

```bash
# transformers
pip install transformers

# torch — pick the build for your hardware. Example for an NVIDIA
# RTX 50-series (Blackwell) GPU, which needs CUDA 12.8 wheels:
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# CPU-only fallback (slow):
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Verify the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The model (~0.4 GB) downloads from the HuggingFace Hub on first use and is
cached afterwards. Override the checkpoint with `MP_DMS_MODEL=<hub-id-or-path>`.

### 3.4 Optional: `vlm` backend (Hugging Face Transformers)

On clusters where Ollama is unavailable, run the VLM backend through
Transformers. The default HF model is
[`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it).

```bash
export MP_VLM_PROVIDER=hf
export MP_VLM_MODEL=google/gemma-4-E4B-it

python -m material_passport.cli \
  --data-root sample_data \
  --output-dir material_ase_lang_output/ \
  --material-methods dms vlm cv \
  --condition-methods vlm rule \
  --material-methods-consensus priority \
  --condition-methods-consensus priority \
  --vlm-provider hf \
  --vlm-model google/gemma-4-E4B-it
```

If the Gemma checkpoint is gated, make sure your HPC environment has access to
your Hugging Face token before the first model download.

### 3.5 Optional: `vlm` backend (local Ollama vision model)

The `vlm` backend asks a local multimodal model served by
[Ollama](https://ollama.com) to choose the material + condition.

```bash
pip install ollama
# install the Ollama server (see ollama.com), then pull a vision model:
ollama pull gemma4
```

Override the model with `MP_VLM_MODEL=<name>` (default `gemma4`).

---

## 4. Running the pipeline

The entry point is `python -m material_passport.cli`. Point it at **one scene**
with `--scene`, or at a **dataset root** of numbered scenes with `--data-root`.

### 4.1 Quick start (offline, no extra deps)

```bash
python -m material_passport.cli --scene sample_data/0
```

### 4.2 Single scene, all three material backends

```bash
python -m material_passport.cli \
  --scene sample_data/0 \
  --output-dir material_ase_lang_output/ \
  --material-methods dms vlm cv \
  --condition-methods vlm rule \
  --material-methods-consensus priority \
  --condition-methods-consensus priority
```

### 4.3 Whole dataset, all three material backends

```bash
python -m material_passport.cli \
  --data-root sample_data \
  --output-dir material_ase_lang_output/ \
  --material-methods dms vlm cv \
  --condition-methods vlm rule \
  --material-methods-consensus priority \
  --condition-methods-consensus priority \
  --skip-existing
```

Add `--visualize` (optionally `--viz-frames N`) to either command to also write
the `material_viz/` step artefacts for each scene.

For multi-scene HPC runs, use process-level parallelism so each worker handles a
different scene:

```bash
python -m material_passport.cli \
  --data-root sample_data \
  --output-dir material_ase_lang_output/ \
  --material-methods dms vlm cv \
  --condition-methods vlm rule \
  --material-methods-consensus priority \
  --condition-methods-consensus priority \
  --skip-existing \
  --num-workers 3 \
  --num-gpus 3 \
  --vlm-provider hf \
  --vlm-model google/gemma-4-E4B-it
```

`--num-workers` controls how many scenes are processed at the same time.
`--num-gpus` binds workers to GPUs round-robin through `CUDA_VISIBLE_DEVICES`.
For large VLMs, start with one worker per GPU.

On a SLURM cluster, `sbatch_material_ase_lang.sh` wraps this command. Submit it
from the repository root and point it at your data:

```bash
mkdir -p logs
sbatch --export=ALL,DATA_ROOT=/path/to/ase/scenes,OUTPUT_DIR=/path/to/output \
  sbatch_material_ase_lang.sh
```

> If you used a dedicated conda env (e.g. `ase_data`) you can call its
> interpreter directly without activating:
> `~/miniconda3/envs/ase_data/bin/python -m material_passport.cli …`

---

## 5. Backends and flags

### Material backends (`--material-methods`)

| Method | Needs | What it does |
| --- | --- | --- |
| `cv`  | nothing | Deterministic heuristic from colour/texture features. Always available; used as the final fallback. |
| `vlm` | Ollama or torch + transformers | Vision-language model classifies the entity crops. |
| `dms` | torch + transformers | SigLIP MINC classifier votes across crops; ≥2 distinct constituent materials are reported as `composite`. |

### Condition backends (`--condition-methods`)

The pipeline infers **two** things per entity. `--condition-methods` selects
which backend(s) estimate the *condition / wear state* (`new`, `good`, `worn`,
`damaged`):

| Method | Needs | What it does |
| --- | --- | --- |
| `rule` | nothing | Heuristic wear score from luminance, colour variation and texture. Always available. |
| `vlm`  | Ollama + model | Reuses the same vision-model call as the material stage. |

There is no `dms` condition backend — the MINC model only classifies material.

### Consensus flags

When you pass more than one method, a consensus strategy decides the winner:

| Flag | Applies to | Choices |
| --- | --- | --- |
| `--material-methods-consensus` | material backends | `priority` (default), `max_confidence`, `vote` |
| `--condition-methods-consensus` | condition backends | `priority` (default), `max_confidence`, `vote` |

* `priority` — take the first method (in priority order) that meets
  `--confidence-threshold`; otherwise fall back to the most confident.
* `max_confidence` — take the single most confident prediction.
* `vote` — majority label, ties broken by summed confidence.

Related options:

* `--priority dms vlm cv` — material method priority order for the `priority`
  strategy (the first available + confident method wins). Default `dms vlm cv`.
* `--confidence-threshold 0.5` — threshold used by the `priority` strategy.
* `--visualize` / `--viz-frames N` — write step images to `material_viz/`.
* `--save-material-frames` — save top-N full RGB frames per entity to
  `material_frames/` for auditing material estimation frame selection.
* `--quiet` — suppress per-scene logs.

---

## 6. Closed material vocabulary

Backends must map their predictions into these per-class vocabularies
(see `material_passport/config.py`):

* **wall**: `painted_plaster`, `wallpaper`, `brick`, `concrete`, `wood_paneling`,
  `tile`, `stone`, `wood`, `metal`, `glass`, `composite`
* **door**: `wood`, `metal`, `glass`, `composite`
* **window** (frame material; glazing is always glass): `wood`, `aluminum`, `pvc`

Condition: `new`, `good`, `worn`, `damaged`.

---

## 7. Troubleshooting

* **"You are sending unauthenticated requests to the HF Hub"** — harmless. After
  the first download the `dms` backend loads from the local cache
  (`local_files_only=True`), so this should not reappear. Set `HF_TOKEN` to
  silence rate-limit notices, or `HF_HUB_OFFLINE=1` to force fully offline.
* **`torch.cuda.is_available()` is False / CUDA arch error** — your `torch`
  build does not match your GPU. Reinstall from the correct wheel index
  (newer GPUs such as the RTX 50-series need the `cu128` index above).
* **`projectaria_tools` import error** — you are not in the env created from
  `environment.yml`. Activate it (or call that env's python directly).
* **A backend is silently skipped** — its dependencies are missing, so the
  pipeline logs `[skip] … unavailable` and falls back. Install the optional
  deps in section 3.
* **`object_instances_to_classes.json` missing** (scenes 20–29) — not required;
  entity↔segmentation association is purely geometric.

---

## License

* **Code:** [MIT](LICENSE).
* **Data** (the Hugging Face release and `sample_outputs/`): derived from ASE,
  non-commercial research only, governed by the
  [Aria Synthetic Environments Dataset License Agreement](https://www.projectaria.com/datasets/ase/license/).
  See [Terms of use](#terms-of-use).

## Citation

If you use this code or the dataset, please cite this repository and the
original ASE dataset:

```bibtex
@misc{naikade2026asematerialpassport,
  author       = {Naikade, Prakash},
  title        = {{ASE Material Passport}: Material and Condition Labels for Aria Synthetic Environments},
  year         = {2026},
  howpublished = {\url{https://github.com/prakashknaikade/ASE_scene_material_lang}},
  note         = {Dataset: \url{https://huggingface.co/datasets/prakashknaikade/ASE-Material-Passport}}
}
```

Attribution: *[Aria Synthetic Environments Dataset, Meta Reality Labs-R]*.
