ARG CUDA_VERSION=12.6.1
ARG HF_TOKEN
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# Basic dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-venv python3-dev \
        build-essential ca-certificates curl wget cmake libnuma-dev nano \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip uv

# Install SGLang
# We copy the dependencies first to cache the installation step (takes 5 min otherwise)
WORKDIR /workspace
COPY ./python/pyproject.toml ./python/pyproject.toml
RUN python3 -m pip install -e "python[all]"

# Now copy the source code and build sglang
COPY . /workspace/sglang
WORKDIR /workspace/sglang/
RUN python3 -m pip install -e "python[all]"

# Install SSSD
WORKDIR /workspace/sglang/sssd_speculator_sglang
RUN python3 -m pip install -e .

WORKDIR /workspace/sglang/

ENV DATA_DIR=data

# Run benchmarks
CMD ["bash", "basic_scripts/run_e2e.sh"]
