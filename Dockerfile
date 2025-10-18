# syntax=docker/dockerfile:1
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
COPY requirements.txt /tmp/pip-tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && PIP_EXTRA_INDEX_URL=${PYTORCH_INDEX_URL} pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

USER vscode
RUN npm config set prefix "${HOME}/.npm-global" \
    && npm install -g @anthropic-ai/claude-code \
    && echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> ~/.bashrc
USER root