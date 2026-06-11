FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install VESTA from local source (baked into image at build time)
COPY src /tmp/vesta-src
COPY pyproject.toml /tmp/vesta-src/pyproject.toml
RUN cd /tmp/vesta-src && pip install --no-cache-dir . && rm -rf /tmp/vesta-src

RUN mkdir -p /app /output

# Bake in the test dataset
COPY harbor/tasks/vesta-distribution-fitting/data /app/data

WORKDIR /app

CMD ["/bin/bash"]
