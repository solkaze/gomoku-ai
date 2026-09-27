# base: 開発用 (devcontainer)。train: 学習用 (依存とバイナリを焼き込んだイメージ)
FROM ubuntu:26.04 AS base

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Locale environment variables
ENV LANG=C.UTF-8

# libssl-dev と pkg-config は ort がビルド時に onnxruntime をダウンロードするのに使う
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Rust (rust-toolchain.toml と同じバージョン)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.98.1

# uv (Python 本体も uv が .python-version に従って入れる)
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /workspace


FROM base AS train

# 依存を先に入れて、ソースを変えただけのときはキャッシュを使う
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY src ./src
RUN cargo build --release --locked \
    # onnxruntime の共有ライブラリは /root/.cache へのシンボリックリンクなので、root 以外でも読めるよう実体に置き換える
    && for f in target/release/libonnxruntime*.so; do cp --remove-destination "$(readlink -f "$f")" "$f"; done

COPY config ./config

# 学習の成果物 (runs/) はボリュームに置く。
# compose ではホストの UID で動かすが、その UID はイメージの /etc/passwd にないので、
# ユーザー名 (torch がキャッシュの場所を決めるのに使う) と書き込み先を環境変数で与える
ENV HOME=/tmp \
    USER=gomoku \
    LOGNAME=gomoku \
    UV_CACHE_DIR=/tmp/uv-cache
CMD ["uv", "run", "--no-sync", "src/train/main.py", "loop"]
