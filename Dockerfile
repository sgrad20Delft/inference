# ---- Base Python image ----
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CFLAGS="-std=c++14 -O3"

# Set working directory
WORKDIR /app

# Install OS-level deps for C++ build & ML tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy repo files ----
COPY . .

# ---- Install MLCommons LoadGen from source ----
# Install pip, wheel, and setup tools
RUN pip install --upgrade pip setuptools wheel

# Then install MLCommons LoadGen manually
RUN git clone --recursive https://github.com/mlcommons/inference.git /tmp/mlcommons-inference \
    && cd /tmp/mlcommons-inference/loadgen \
    && python3 -m pip install . \
    && rm -rf /tmp/mlcommons-inference

# Then install your own project
RUN pip install -e .

RUN git clone --recursive https://github.com/mlcommons/inference.git /tmp/mlcommons-inference \
    && cd /tmp/mlcommons-inference/loadgen \
    && python3 -m pip install . \
    && rm -rf /tmp/mlcommons-inference

# ---- Install Python package + dependencies ----
RUN pip install --upgrade pip && pip install -e .

# ---- Default entry (optional) ----
# ENTRYPOINT ["vision-infer"]
