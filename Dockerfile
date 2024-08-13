FROM nvidia/cuda:12.3.0-base-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt-get install -y locales
RUN apt-get install -y python3.9 python3-pip
RUN apt install -y git-all
RUN apt install -y wget
RUN apt install -y curl
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install pandas
RUN pip install ultralytics
RUN pip install matplotlib
RUN pip install albumentations
RUN pip install tqdm
RUN pip install tensorflow==2.15.1
RUN pip install tensorflow[and-cuda]
RUN pip install torch==2.3.1
RUN pip install torchvision==0.18.1
RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk

RUN mkdir src

# Install git, supervisor, VNC, & X11 packages
RUN set -ex; \
    apt-get update; \
    apt-get install -y \
      bash \
      fluxbox \
      git \
      net-tools \
      novnc \
      supervisor \
      x11vnc \
      xterm \
      xvfb

# Setup demo environment variables
ENV HOME=/root \
    DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    DISPLAY=:0.0 \
    DISPLAY_WIDTH=1024 \
    DISPLAY_HEIGHT=768 \
    RUN_XTERM=yes \
    RUN_FLUXBOX=yes