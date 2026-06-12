FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install VESTA from local source (baked into image at build time).
COPY src /tmp/vesta-src/src
COPY pyproject.toml /tmp/vesta-src/pyproject.toml
COPY README.md /tmp/vesta-src/README.md
COPY LICENSE /tmp/vesta-src/LICENSE
RUN cd /tmp/vesta-src \
    && uv pip install --system --no-cache . \
    && rm -rf /tmp/vesta-src

RUN mkdir -p /app /output

# Bake in the task dataset.
COPY harbor/tasks/vesta-distribution-fitting/data /app/data

WORKDIR /app

CMD ["/bin/bash"]
