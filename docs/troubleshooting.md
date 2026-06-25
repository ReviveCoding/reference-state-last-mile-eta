# Troubleshooting

## Torch installation is slow or unavailable

The base package does not require torch. The advanced HSG smoke path requires the optional `advanced`/`gpu` dependency or an environment where torch is available. If torch is unavailable, run the CORE smoke and package gates first, then install the optional dependency explicitly before `train-gpu`.

## `pip check` reports unrelated global packages

Run qualification in a fresh virtual environment without `--system-site-packages`. Global environment conflicts, such as unrelated notebook/media packages, must not be counted as project dependency failures unless they are present in the clean project environment.

## API benchmark startup fails

Check the preserved Uvicorn log tail in the raised error. The benchmark preserves the caller's `PYTHONPATH` and prepends `src` so optional runtime paths are not accidentally hidden.

## Clean candidate manifest mismatch

Regenerate the repository manifest after source, report, or generated qualification files change:

```bash
python scripts/tasks.py release-manifest
python scripts/tasks.py clean-candidate
python scripts/tasks.py release-manifest
```
