FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL maintainer="Matt Bowring <mbowring@purdue.edu>"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    libboost-all-dev \
    libeigen3-dev \
    bison \
    flex \
    libfl-dev \
    tcl \
    libcairo2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir \
    "pyunpack>=0.1.2" \
    "patool>=1.12" \
    "matplotlib>=2.2.2" \
    "cairocffi>=0.9.0" \
    "pkgconfig>=1.4.0" \
    "setuptools>=39.1.0" \
    scipy \
    shapely \
    ncg_optimizer \
    torch-optimizer

WORKDIR /workspace
