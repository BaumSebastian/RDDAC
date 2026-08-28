# Publishing the RDDAC teaser

Publishes the ~174 MB sample (manifest + CSV + 18 experiments + docs) to **Kaggle**
and **Hugging Face**, plus the tutorial notebooks as **Kaggle notebooks**. The full
~87 GB dataset stays on DaRUS; this is the discovery teaser.

The scripts here are committed and run in CI (`.github/workflows/publish.yml`);
only `.staging/` is git-ignored. After the one-time setup below, releases keep
the public surfaces fresh automatically:

| Release tag | PyPI | Kaggle kernels | Kaggle dataset | Hugging Face | Zenodo |
|---|---|---|---|---|---|
| patch/minor (e.g. `1.0.1`) | ✓ | ✓ (regenerated) | — | — | — |
| major (e.g. `2.0.0`) | ✓ | ✓ | ✓ (one DaRUS fetch, shared) | ✓ | ✓ (new version) |

Order on a major: publish the new version on **DaRUS first**, then push the tag —
the CI fetch reads DaRUS anonymously at tag time.

## Manual upload commands (first publish / ad hoc)

```bash
# from the repo root; ./data must hold the three core files (or set RDDAC_TEASER_SRC)
# 1. Kaggle dataset  (first time: use `create` instead of `version`)
./publish/kaggle/upload.sh create

# 2. Kaggle notebooks -> the dataset's Code tab
./publish/kaggle/kernels/push.sh

# 3. Hugging Face dataset
HF_REPO=BaumSebastian/rddac-teaser ./publish/huggingface/upload.sh

# 4. Zenodo (first time: creates the record as a DRAFT to review on the site;
#    afterwards: write the printed conceptrecid into publish/zenodo/metadata.json)
ZENODO_API_TOKEN=... python publish/zenodo/upload.py
```

That's it. Re-run any of the three to update that target.

## One-time setup

- **Kaggle CLI**: `pip install kaggle`; auth is ambient via `~/.kaggle/access_token`
  (locally) or the `KAGGLE_API_TOKEN` secret in the `kaggle` environment (CI).
- **Hugging Face CLI**: `hf auth login` with a **write** token
  (huggingface.co/settings/tokens) — locally; CI uses the `HF_TOKEN` secret in
  the `huggingface` environment.
- **Data**: defaults to `./data`; override with `RDDAC_TEASER_SRC=<data source dir>`
  to stage from wherever the bundle lives (a dir with `metadata.json`,
  `process_parameters.csv`, `h5/sample.zip`, and ideally `rddac_documentation.pdf`;
  without the PDF there, it is taken from `.helper/v1/documentation/`).

## How each upload works

**`stage_teaser.sh`** builds `publish/.staging/` from `$RDDAC_TEASER_SRC` (default `./data`):
`data/{metadata.json, process_parameters.csv, h5/sample.zip}` + `rddac_documentation.pdf`
+ `README.md`. All three uploaders call it first.

| Command | What it uploads | Notes |
|---|---|---|
| `kaggle/upload.sh {create,version}` | staged files + `dataset-metadata.json` | `--dir-mode zip`; Kaggle **auto-extracts** the h5 zip. Page **description + column docs come from `dataset-metadata.json`**, not the README. `version` first pulls current metadata to avoid the "non current" error. |
| `kaggle/kernels/push.sh` | the tutorial notebooks as kernels | `build.py` adapts each (pip install, `rddac download --small`, 87 GB CTA) and attaches the dataset. |
| `huggingface/upload.sh` | staged files + `notebooks/` + card | Card = `card-header.md` (YAML) + `teaser/README.md`. HF does **not** extract zips, so `rddac.load` works off the repo. |

## Editing content (single source)

- **README body** (both cards + the bundled file): `teaser/README.md` — edit once,
  all surfaces update on the next push.
- **HF card YAML** (license, tags): `huggingface/card-header.md`.
- **Kaggle page text** (title, subtitle, description, tags, CSV column docs):
  `kaggle/dataset-metadata.json`.
- **Notebook adaptations**: `kaggle/kernels/build.py` (notebooks themselves stay
  the source of truth in `../notebooks/`).

## Visibility

Kaggle dataset + kernels are made public on the **site** (a public kernel needs a
public dataset first). HF dataset repos are public by default.

## Croissant registry (mlcommons/croissant)

See `croissant-registry/README.md` — a prepared PR adds the DDACS and RDDAC
manifests to the community gallery at
https://github.com/mlcommons/croissant/tree/main/datasets/1.1.
