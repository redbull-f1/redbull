# python - <<'PY'
import torch, os, sys
print("torch __version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)   # None이면 CPU빌드임
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

