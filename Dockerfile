# syntax=docker/dockerfile:1.4
FROM rust:1.88-slim-bookworm AS base

# 1) Install Python and system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential curl pkg-config libssl-dev protobuf-compiler libprotobuf-dev docker.io git openssh-client autossh \
 && rm -rf /var/lib/apt/lists/*

# 2) Install the 'uv' CLI
ARG UV_VERSION=0.11.26
RUN pip install --break-system-packages "uv==$UV_VERSION"

WORKDIR /app

# 3) Copy dependency descriptors
COPY pyproject.toml uv.lock ./

# 4) Create venv and sync dependencies.
ENV VENV_DIR=/opt/venv
ENV VIRTUAL_ENV=$VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"
RUN uv venv --python python3 $VENV_DIR \
 && uv sync --active --frozen --no-install-project

# 5) Copy application code and install it
COPY . .
RUN uv pip install --no-deps -e .

ENTRYPOINT ["af"]
