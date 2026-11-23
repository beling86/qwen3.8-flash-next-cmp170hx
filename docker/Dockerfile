FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

ENV VLLM_WORKDIR=/workspace/vllm
ENV VLLM_SOURCE=/workspace/vllm/src
ENV VLLM_VENV=/workspace/vllm/.venv
ENV PATH="$VLLM_VENV/bin:/usr/local/cuda/bin:$PATH"

WORKDIR /workspace

# Create optimization directory and virtual environment
RUN mkdir -p /workspace/optimization "$VLLM_WORKDIR" && \
    uv venv --python 3.12 "$VLLM_VENV"

# Copy local dependencies and wheels
COPY vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl /workspace/
COPY repo/requirements/tested-constraints.txt /workspace/tested-constraints.txt
COPY ple_ssd_io.so /workspace/optimization/ple_ssd_io.so

# Copy patched vllm source tree and apply DFlash2 + hybrid KV packing patches
COPY vllm-src /workspace/vllm/src
COPY repo/docker/patch_dflash2.py /tmp/patch_dflash2.py
RUN python3 /tmp/patch_dflash2.py && rm -f /tmp/patch_dflash2.py

# Install precompiled vLLM wheel and requirements using cu130 torch backend
RUN uv pip install --python "$VLLM_VENV/bin/python" --torch-backend=cu130 \
    -c /workspace/tested-constraints.txt \
    /workspace/vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl

RUN uv pip install --python "$VLLM_VENV/bin/python" --torch-backend=cu130 \
    -c /workspace/tested-constraints.txt \
    -r "$VLLM_SOURCE/requirements/build/cuda.txt" 'huggingface-hub==1.32.0'

# Install patched vLLM in editable mode using precompiled wheel extensions
ENV VLLM_USE_PRECOMPILED=1
ENV VLLM_VERSION_OVERRIDE=0.29.1rc1.dev402+ga5a30471f.ple1
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=/workspace/vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl

RUN cd "$VLLM_SOURCE" && \
    uv pip install --python "$VLLM_VENV/bin/python" --reinstall \
    --config-settings editable_mode=compat --no-deps --no-build-isolation -e .

# Clean up wheel to save image space
RUN rm -f /workspace/vllm-0.29.1rc1.dev402+ga5a30471f-cp38-abi3-manylinux_2_28_x86_64.whl

WORKDIR /workspace
ENTRYPOINT ["vllm"]
