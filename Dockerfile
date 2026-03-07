# Listenr — AMD ROCm fine-tuning image
#
# Base: official AMD-tested ROCm 7.2 + PyTorch 2.9.1 image (Python 3.12).
# Ref:  https://rocm.docs.amd.com/en/latest/how_to/pytorch_install/pytorch_install.html
#
# Pull:  podman pull rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
# Build: podman build -t listenr-rocm .
# Run:   see docs/finetune-amd.md
#
# NOTE: sounddevice (microphone capture) will not work inside this container.
#       This image is intended for fine-tuning only (listenr-finetune /
#       listenr-build-dataset), not real-time audio capture.
#
# NOTE: listenr requires Python >=3.13 for local installs; this image uses
#       Python 3.12 (the AMD-tested version). --ignore-requires-python is safe
#       here — the codebase uses no 3.13-only syntax.

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

# ── system packages ──────────────────────────────────────────────────────────
# libsndfile1   : required by soundfile (audio I/O in finetune data pipeline)
# ffmpeg        : optional but useful for converting audio files
#
# IMPORTANT: We must not upgrade libdrm, mesa, or any ROCm library — doing so
# breaks the GPU stack that ships with the base image. Use --no-upgrade to
# install only what is missing (both packages are typically absent from the
# base image but their deps like libdrm are already present).
RUN apt-get update && apt-get install -y --no-install-recommends --no-upgrade \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── project install ──────────────────────────────────────────────────────────
WORKDIR /app
COPY . /app

# Freeze the ROCm-aware torch/torchvision/torchaudio/triton that ship in the
# base image before installing finetune extras. Without this, pip resolves
# transformers' torch dependency and pulls a CPU-only build from PyPI.
# pip show works regardless of whether torch was installed via URL or wheel.
RUN pip show torch torchvision torchaudio triton 2>/dev/null \
    | awk '/^Name:/{name=$2} /^Version:/{print name "==" $2}' \
    > /tmp/torch-constraints.txt \
    && cat /tmp/torch-constraints.txt

# Install core + finetune extras, pinning torch to the ROCm version above.
# --ignore-requires-python: base image is Python 3.12; constraint is >=3.13.
RUN pip install --no-cache-dir \
        --ignore-requires-python \
        --constraint /tmp/torch-constraints.txt \
        -e ".[finetune]"

# ── runtime defaults ─────────────────────────────────────────────────────────
# Pin to GPU 0 by default to avoid imbalance crashes on multi-GPU systems.
# Override at runtime: -e HIP_VISIBLE_DEVICES=0,1
#
# Do NOT set HSA_OVERRIDE_GFX_VERSION here — an empty string is not the same
# as unset and causes ROCm to fail. Set it at runtime only if your GPU needs
# it (e.g. -e HSA_OVERRIDE_GFX_VERSION=10.3.0 for RX 6000 series).
ENV HIP_VISIBLE_DEVICES="0"

CMD ["listenr-finetune", "--help"]
