# GPU execution

Use Python 3.11 and install the CUDA-enabled PyTorch build appropriate for the local driver through
the official PyTorch installation selector. Verify `torch.cuda.is_available()` before training.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
make train-gpu
```

For a normalized full LaDe file:

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/gpu_full.example.yaml \
  --mode all
```

Preserve the following before making GPU-performance claims:

- CUDA/PyTorch/hardware manifest
- AMP enabled flag
- epoch time and examples/second
- peak allocated/reserved VRAM
- FP32 versus AMP quality comparison
- seed and checkpoint hash
- failure/OOM fallback log
