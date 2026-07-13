# Registering DDACS + RDDAC in the Croissant gallery (manual PR)

The mlcommons/croissant repo hosts a community gallery of Croissant datasets:
one folder per dataset under `datasets/<spec-version>/<name>/metadata.json`
(https://github.com/mlcommons/croissant/tree/main/datasets). Both our manifests
are Croissant 1.1, so they go under `datasets/1.1/`.

This stays a manual PR by design: majors are rare and the PR needs a human for
review anyway.

## One-time

1. Sign the MLCommons Association CLA: https://mlcommons.org/community/subscribe/
   (required before any PR can be merged).
2. Fork https://github.com/mlcommons/croissant.

## Per major release

```bash
# in your fork, branch from main
git checkout -b add-ddacs-rddac

mkdir -p datasets/1.1/ddacs datasets/1.1/rddac
cp <DDACS repo>/docs/metadata.json  datasets/1.1/ddacs/metadata.json
cp <RDDAC repo>/docs/metadata.json  datasets/1.1/rddac/metadata.json

# validate like their CI does
pip install mlcroissant
python -c "import mlcroissant as mlc; mlc.Dataset('datasets/1.1/ddacs/metadata.json'); mlc.Dataset('datasets/1.1/rddac/metadata.json'); print('both valid')"

git add datasets/1.1 && git commit -m "Add DDACS and RDDAC datasets (Croissant 1.1)"
git push -u origin add-ddacs-rddac
# open the PR against mlcommons/croissant main
```

Suggested PR text: two published sheet-metal-forming datasets from the
University of Stuttgart / DaRUS — DDACS (32,466 FEM simulations,
doi:10.18419/DARUS-4801) and RDDAC (9,000 physical experiments,
doi:10.18419/DARUS-5589), simulation/experiment counterparts sharing one
schema style; manifests are the ones published with the datasets and consumed
by the `ddacs` / `rddac` PyPI packages.

Snapshots of both manifests as of this preparation sit next to this README
(`ddacs-metadata.json`, `rddac-metadata.json`) — but copy fresh ones from the
repos' `docs/metadata.json` when you actually open the PR.
