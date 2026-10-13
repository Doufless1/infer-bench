# infer-bench in a container.
#
# The CI workflow (.github/workflows/ci.yml) builds this image and runs a short benchmark inside it on every push.
#
# Usage:
#   docker run --rm infer-bench selfcheck
#   docker run --rm -v "$PWD/models:/work/models" -v "$PWD/results:/work/results" infer-bench \
#       run models/model.onnx --label demo --out results/demo.json
#   docker run --rm -p 8050:8050 -v "$PWD/results:/work/results" infer-bench \
#       dashboard --host 0.0.0.0 --results results
#
# Timing inside a container is noisier than on the host, and CPU limits change the numbers. Power counters
# (RAPL) are normally not visible in a container: use --power-watts, or run natively, for power figures.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-lock.txt pyproject.toml README.md LICENSE ./
COPY infer_bench ./infer_bench
RUN pip install -r requirements-lock.txt && pip install --no-deps .

# Models and results live in a mounted folder, not in the image.
WORKDIR /work
ENTRYPOINT ["infer-bench"]
CMD ["--help"]
