# Temper's engine as a server: `temper serve` with the browser interface at /ui/, chat through llama.cpp,
# and everything Temper keeps (models, chats, runs, exports, settings) in the /data volume.
#
#   docker build -t temper .
#   docker run -d -p 8765:8765 -v temper:/data --name temper temper
#   docker logs temper        # prints the link to open, with the key
#
# The image runs models on the CPU. Training needs the GPU profile, which isn't in this image yet.

# --- The interface ------------------------------------------------------------------------------
FROM node:22-trixie-slim AS ui
WORKDIR /src
COPY package.json package-lock.json .npmrc ./
RUN npm ci --no-audit --no-fund
COPY index.html tsconfig.json vite.config.ts ./
COPY public public
COPY src src
RUN npm run build

# --- The engine ---------------------------------------------------------------------------------
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /usr/local/bin/uv

# llama.cpp's Linux builds need OpenMP and libcurl; ca-certificates for downloads from Hugging Face.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 temper \
    && mkdir /data && chown temper:temper /data

WORKDIR /opt/temper
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY engine/pyproject.toml engine/uv.lock engine/
RUN cd engine && uv sync --frozen --no-dev --no-install-project
COPY engine/temper_engine engine/temper_engine
RUN cd engine && uv sync --frozen --no-dev --no-editable

# The pinned llama.cpp build, downloaded and checked against its SHA-256 by the engine's own installer,
# kept outside /data so a volume doesn't hide it; the engine finds it on PATH.
RUN TEMPER_HOME=/tmp/seed engine/.venv/bin/python -c "from temper_engine import llamacpp; print(llamacpp.install(backend='cpu'))" \
    && bin=$(dirname "$(find /tmp/seed/llama.cpp -name llama-server -type f | head -1)") \
    && mv "$bin" /opt/llama.cpp \
    && chmod -R a+rX /opt/llama.cpp \
    && rm -rf /tmp/seed \
    && /opt/llama.cpp/llama-server --version

COPY --from=ui /src/dist /opt/temper/ui
RUN printf '#!/bin/sh\nexec /opt/temper/engine/.venv/bin/temper-engine "$@"\n' > /usr/local/bin/temper && chmod +x /usr/local/bin/temper

ENV PATH="/opt/llama.cpp:$PATH" \
    TEMPER_HOME=/data \
    HF_HOME=/data/huggingface \
    TEMPER_UI=/opt/temper/ui \
    PYTHONUNBUFFERED=1
USER temper
VOLUME /data
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health')"
ENTRYPOINT ["temper"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
