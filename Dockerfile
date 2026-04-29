# syntax=docker/dockerfile:1.4
FROM rust:1.88-slim-bookworm AS base

# 1) Install Python and system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential curl pkg-config libssl-dev protobuf-compiler libprotobuf-dev docker.io git openssh-client \
 && rm -rf /var/lib/apt/lists/*

# 2) Install the 'uv' CLI
RUN pip install --break-system-packages uv

WORKDIR /app

# 3) Copy dependency descriptors
COPY pyproject.toml uv.lock ./

# 4) Create venv and sync dependencies in one step.
#    After sync, replace py-scale-codec (`scalecodec`) with cyscale: the two
#    packages share the `scalecodec` import namespace, and async-substrate-interface
#    >=2.0 raises at import time if `scalecodec` distribution metadata is present.
#    bittensor's transitive deps still pin `scalecodec`, so we swap it out here.
ENV VENV_DIR=/opt/venv
ENV VIRTUAL_ENV=$VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"
RUN uv venv --python python3 $VENV_DIR \
 && uv sync \
 && uv pip uninstall scalecodec \
 && uv pip install --force-reinstall --no-cache cyscale

# 5) Copy application code and install it
COPY . .
RUN uv pip install -e .

ENTRYPOINT ["af"]
