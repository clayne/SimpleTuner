# SimpleTuner needs CU141
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# /workspace is the default volume for Runpod & other hosts
WORKDIR /workspace

# Prevents different commands from being stuck by waiting
# on user input during build
ENV DEBIAN_FRONTEND noninteractive

# Install misc unix libraries
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
	openssh-server \
	openssh-client \
	git \
	git-lfs \
	wget \
	curl \
	tmux \
	tldr \
	nvtop \
	vim \
	rsync \
	net-tools \
	less \
	iputils-ping \
	7zip \
	zip \
	unzip \
	htop \
	inotify-tools \
	libgl1-mesa-glx \
	libsm6 \
	libxext6 \
	ffmpeg \
	python3 \
	python3-pip \
	python3.10-venv

# Fix nvtop (may segfault with older versions)
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update -y \
    && apt-get install -y --no-install-recommends software-properties-common \
    && apt-add-repository ppa:flexiondotorg/nvtop \
    && apt-get install -y --no-install-recommends nvtop \
    && apt-get remove -y software-properties-common

# Set up git to support LFS, and to store credentials; useful for Huggingface Hub
RUN git lfs install && git config --global credential.helper store

# Python
RUN --mount=type=cache,target=/root/.cache python3 -m pip install pip --upgrade

# HF
ENV HF_HOME=/workspace/huggingface
RUN --mount=type=cache,target=/root/.cache pip3 install "huggingface_hub[cli]"

# WanDB
RUN --mount=type=cache,target=/root/.cache pip3 install wandb

# Clone SimpleTuner
#RUN git clone https://github.com/bghira/SimpleTuner --branch release
RUN git clone https://github.com/bghira/SimpleTuner --branch main # Uncomment to use latest (possibly unstable) version

# Install SimpleTuner
RUN --mount=type=cache,target=/root/.cache pip3 install poetry
RUN --mount=type=cache,target=/root/.cache cd SimpleTuner && python3 -m venv .venv && poetry install --no-root
RUN chmod +x SimpleTuner/train.sh

# Copy start script with exec permissions
COPY --chmod=755 docker-start.sh /start.sh

# Ensure SSH access. Not needed for Runpod but is required on Vast and other Docker hosts
EXPOSE 22/tcp

# Dummy entrypoint
ENTRYPOINT [ "/start.sh" ]
